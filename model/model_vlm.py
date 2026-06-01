import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        super().__init__(**kwargs)

class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)

# 继承自语言模型
class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p32-256-ve"):
        self.config = config or VLMConfig()
        super().__init__(self.config)
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)

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
        """将输入的图像数据转换为适合输入到视觉模型的格式。
        Args:   image (PIL.Image or np.ndarray): 输入的图像数据，可以是PIL图像对象或NumPy数组。
                processor : 用于预处理图像数据的处理器对象，通常是从预训练视觉模型中获取的。
        Returns:     dict: 包含预处理后图像数据的字典，通常包含键 'pixel_values'，其值是一个张量，形状为 (batch_size, num_channels, height, width)，适合输入到视觉模型中进行特征提取。
        """
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        """将输入的图像数据转换为视觉特征张量。
        Args:   image_inputs (torch.Tensor or dict): 输入的图像数据，可以是一个张量或一个包含图像数据的字典。每个图像数据应该已经过预处理并转换为适合输入到视觉模型的格式。
                vision_model (nn.Module): 用于提取视觉特征的预训练视觉模型。
        Returns:     torch.Tensor: 提取的视觉特征张量，形状为 (batch_size, num_patches, hidden_size)，其中包含了每个输入图像的视觉特征表示。"""
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        """将视觉特征插入到语言模型的隐藏状态中，替换掉输入序列中对应的特殊标记位置。
        Args:     tokens (torch.Tensor): 输入的token ID张量，形状为 (batch_size, seq_length)。
            h (torch.Tensor): 语言模型的隐藏状态张量，形状为 (batch_size, seq_length, hidden_size)。
            vision_tensors (torch.Tensor, optional): 视觉特征张量，形状为 (batch_size, image_token_len, hidden_size)。默认为None。      
        Returns:     torch.Tensor: 更新后的隐藏状态张量，形状为 (batch_size, seq_length, hidden_size)，其中包含了视觉特征。"""
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                sample_val = next(iter(pixel_values.values()))
                if sample_val.ndim == 5:
                    bs, num = sample_val.shape[:2]
                    vision_tensors = self.vision_proj(MiniMindVLM.get_image_embeddings({k: v.flatten(0, 1) for k, v in pixel_values.items()}, self.vision_encoder)).view(bs, num, self.config.image_token_len, -1)
                else:
                    vision_tensors = self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder))
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape
                vision_tensors = torch.stack([self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)) for i in range(num)], dim=1)
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=input_ids.shape[1])

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # dummy gradient for DDP
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)