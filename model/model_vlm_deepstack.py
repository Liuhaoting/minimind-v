import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Tuple, List, Union
from .model_minimind import MiniMindConfig, MiniMindForCausalLM, precompute_freqs_cis, MOEFeedForward
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')

# VLMDeepStackConfig
class VLMDeepStackConfig(MiniMindConfig):
    """
    DeepStack VLM 配置
    - deepstack_visual_indexes: 从视觉编码器的哪些中间层抽取特征（默认 [3,7,11]，共 3 层）
    """
    model_type = "minimind-v-deepstack"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        self.deepstack_visual_indexes = kwargs.get("deepstack_visual_indexes", [3, 7, 11])
        super().__init__(**kwargs)

#  MMVisionProjector 
class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)

#  MiniMindDeepStackVLM
class MiniMindDeepStackVLM(MiniMindForCausalLM):
    config_class = VLMDeepStackConfig

    def __init__(self, config: VLMDeepStackConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):
        self.config = config or VLMDeepStackConfig()
        super().__init__(self.config)
        # 视觉编码器（冻结
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)

        # 视觉编码器层数
        try:
            self._vision_num_layers = len(self.vision_encoder.vision_model.encoder.layers)
        except Exception:
            self._vision_num_layers = 12  

        # 过滤掉越界的索引，且不超过 LLM 层数
        self.deepstack_visual_indexes = [
            i for i in self.config.deepstack_visual_indexes
            if i < self._vision_num_layers
        ][:self.config.num_hidden_layers]
        n_deep = len(self.deepstack_visual_indexes)

        # 每个抽取层对应一个独立 merger
        self.deepstack_merger_list = nn.ModuleList([
            MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size)
            for _ in range(n_deep)
        ])

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        try:
            model = SiglipVisionModel.from_pretrained(model_path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(model_path)
        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs

    #  将图片逐层过视觉编码器 → 在指定层抽取中间特征 → 各自投影 → 返回
    def _forward_vision_deepstack(self, pixel_values):
        """
        参数:
            pixel_values: (bs, C, H, W)
        返回:
            deep_feats: list of (bs, image_token_len, hidden_size),
                        len = n_deep, 每项是中间层特征经 merger 投影后的结果
        """
        device = next(self.vision_encoder.parameters()).device
        pixel_values = pixel_values.to(device)

        hidden_states = self.vision_encoder.vision_model.embeddings(pixel_values)

        collected = []
        for layer_idx, blk in enumerate(self.vision_encoder.vision_model.encoder.layers):
            hidden_states = blk(hidden_states)[0]
            if layer_idx in self.deepstack_visual_indexes:
                collected.append(hidden_states.clone())

        deep_feats = []
        for i, feat in enumerate(collected):
            feat = self.vision_encoder.vision_model.post_layernorm(feat)
            feat = self.deepstack_merger_list[i](feat)
            deep_feats.append(feat)

        return deep_feats

    #  从 input_ids 构建 bool mask：True 的位置是 <|image_pad|>
    @torch.no_grad()
    def _build_visual_mask(self, input_ids):
        marker = self.config.image_ids[0]
        return (input_ids == marker)  # (bs, seq_len)

    # 前 n_deep 层 = TransformerBlock + deepstack 注入，后续层为标准 TransformerBlock
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        _, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Token Embedding 
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 视觉特征提取（仅首次前向） 
        deepstack_visual_embeds = None 
        visual_pos_masks = None 
        n_deep = len(self.deepstack_visual_indexes) # 3 

        if pixel_values is not None and start_pos == 0:
            # 解析 pixel_values 为统一张量（兼容 dict / 5D / 6D）
            if hasattr(pixel_values, 'keys'):
                sample_val = next(iter(pixel_values.values()))
                if sample_val.ndim == 5:
                    pixel_tensor = {k: v.flatten(0, 1) for k, v in pixel_values.items()}
                    pixel_tensor = pixel_tensor.get('pixel_values', next(iter(pixel_tensor.values())))
                else:
                    pixel_tensor = pixel_values.get('pixel_values', next(iter(pixel_values.values())))
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                pixel_tensor = pixel_values

            # 扁平化 5D → 4D
            num = 1 # 图片数
            if pixel_tensor.dim() == 5:
                num = pixel_tensor.shape[1]
                pixel_tensor = pixel_tensor.flatten(0, 1)

            # DeepStack 视觉编码器提取各层中间特征
            deep_feats = self._forward_vision_deepstack(pixel_tensor)  # list of (bs, 64, hidden_size)

            # 恢复 batch 维度（多图场景：多图时只取第一张）
            if num > 1:
                actual_bs = deep_feats[0].size(0) // num
                deep_feats = [f.view(actual_bs, num, *f.shape[1:])[:, 0] for f in deep_feats]

            # 准备 mask 和注入特征
            visual_pos_masks = self._build_visual_mask(input_ids)           # (bs, seq_len)
            deepstack_visual_embeds = deep_feats[:n_deep]                  # 3 组注入特征

            # 将 image_pad 位置清零（不保留原始 embedding，由各层注入填补）
            hidden_states = hidden_states.masked_fill(visual_pos_masks.unsqueeze(-1), 0.0)

        # RoPE旋转位置编码
        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling,)
            self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length],
        )

        # Transformer 层循环 
        presents = []
        for layer_idx, (layer, past_kv) in enumerate(zip(self.model.layers, past_key_values)):

            # DeepStack 注入：前 n_deep 层，在 Transformer 之前把视觉特征加到 image 位置
            if (deepstack_visual_embeds is not None and visual_pos_masks is not None  and layer_idx < n_deep):

                ve = deepstack_visual_embeds[layer_idx]  # (bs, n_patches, hidden_size)
                # 展平 visual_embeds 匹配 mask 索引
                ve_flat = ve.reshape(-1, ve.size(-1))     # (bs * n_patches, hidden_size)
                hidden_part = hidden_states[visual_pos_masks]  # (bs * n_patches, hidden_size)
                # 加法注入
                hidden_states = hidden_states.clone()
                hidden_states[visual_pos_masks] = hidden_part + ve_flat # (bs, n_patches, hidden_size)

            # 标准 Transformer Block
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_kv,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        # 后处理
        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward))

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100,)

        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents,
                                        hidden_states=hidden_states)

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {
                    k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1)))
                    for k, v in pv.items()
                }
            else:
                kwargs['pixel_values'] = pv.repeat(
                    num_return_sequences, *([1] * (pv.ndim - 1))
                )
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
