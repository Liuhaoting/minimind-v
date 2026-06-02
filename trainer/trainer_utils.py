"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist # 引入 PyTorch 的分布式通信模块，用于多GPU训练中的通信和同步
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM


def get_model_params(model, config, ignore_patterns=['vision_encoder']):
    """计算并打印模型的参数量，区分专家参数和基础参数。
    Args:    model: 待计算参数量的模型对象。
            config: 模型配置对象，包含专家数量、每个token的专家数量等信息。
            ignore_patterns: 在计算参数量时需要忽略的参数名称模式列表，默认为包含'vision_encoder'，即不计算视觉编码器的参数量。
    """
    def should_count(n): 
        return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'模型参数量: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'模型参数量: {total:.2f}M')


def is_main_process():
    """判断当前进程是否为主进程，通常在分布式训练中使用，以确保只有主进程执行某些操作"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """在分布式训练中，只有主进程会打印日志内容，避免重复输出"""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """根据当前训练步骤和总训练步骤计算学习率，使用余弦退火调度策略。lr从初始值逐渐衰减到0.1倍初始值。
    Args:    current_step: 当前的训练步骤数。
            total_steps: 总的训练步骤数。
            lr: 初始学习率。
    """
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    
    dist.init_process_group(backend="nccl") # 初始化分布式环境，使用NCCL后端进行GPU通信
    local_rank = int(os.environ["LOCAL_RANK"]) # 获取当前进程的本地GPU编号，通常由分布式训练框架自动设置
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed) # 固定 Python 内置 random 模块的种子。
    np.random.seed(seed) # 固定 NumPy 的随机种子。
    torch.manual_seed(seed) # 固定 PyTorch CPU 的随机种子。
    torch.cuda.manual_seed(seed) # 固定 PyTorch 当前 GPU 的随机种子。
    torch.cuda.manual_seed_all(seed) # 固定 PyTorch 所有 GPU 的随机种子。
    torch.backends.cudnn.deterministic = True # 设置 cuDNN 后端为确定性模式，确保每次运行得到相同的结果。
    torch.backends.cudnn.benchmark = False # 禁用 cuDNN 的自动优化功能，进一步确保结果的可复现性。


def init_vlm_model(vlm_config, from_weight='pretrain_vlm', tokenizer_path='../model', vision_model_path='../model/siglip2-base-p32-256-ve', save_dir='../out', device='cuda', freeze_llm=0):
    """初始化视觉语言模型，加载预训练权重，并根据冻结策略设置参数的可训练性。
    Args:    vlm_config: 视觉语言模型的配置对象，包含模型结构和超参数等信息。
            from_weight: 预训练权重的标识字符串，默认为'pretrain_vlm'，用于指定加载哪个预训练权重文件。           tokenizer_path: 预训练分词器的路径，默认为'../model'，用于加载分词器。
            vision_model_path: 预训练视觉模型的路径，默认为'../model/siglip2-base-p32-256-ve'，用于加载视觉编码器和处理器。
            save_dir: 模型权重保存的目录，默认为'../out'，用于保存和加载模型权重文件。
            device: 设备类型，默认为'cuda'，用于指定模型加载到哪个设备上。
            freeze_llm: 冻结语言模型的策略，默认为0，表示不冻结语言模型参数。其他值表示不同的冻结策略，具体实现见代码。
    Returns:    model: 初始化后的视觉语言模型对象，已经加载预训练权重并根据冻结策略设置了参数的可训练性。
            tokenizer: 加载的分词器对象，用于文本数据的预处理。
            preprocess: 加载的视觉处理器对象，用于图像数据的预处理。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindVLM(vlm_config, vision_model_path=vision_model_path)
    
    if from_weight != 'none':
        moe_suffix = '_moe' if vlm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    
    # 1、全部冻结，只打开vision_proj梯度
    for name, param in model.named_parameters():
        if 'vision_proj' not in name:
            param.requires_grad = False

    # 2、判断策略
    if freeze_llm == 0:
        for name, param in model.named_parameters():
            if 'vision_encoder' not in name:
                param.requires_grad = True
    elif freeze_llm == 1:
        last_idx = vlm_config.num_hidden_layers - 1
        for name, param in model.model.named_parameters():
            if 'layers.0.' in name or f'layers.{last_idx}.' in name:
                param.requires_grad = True
    elif freeze_llm == 2:
        pass

    get_model_params(model, vlm_config)
    Logger(f'可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vlm_checkpoint(vlm_config, weight='pretrain_vlm', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """保存或加载视觉语言模型的检查点，包括模型权重、优化器状态、训练进度等信息。
    Args:    vlm_config: 视觉语言模型的配置对象，包含模型结构和超参数等信息。
            weight: 预训练权重的标识字符串，默认为'pretrain_vlm'，用于指定保存或加载哪个预训练权重文件。
            model: 视觉语言模型对象，如果提供则保存模型权重；如果为None则尝试加载模型权重。
            optimizer: 优化器对象，如果提供则保存优化器状态；如果为None则不保存优化器状态。
            epoch: 当前的训练轮数，默认为0，用于保存训练进度。
            step: 当前的训练步数，默认为0，用于保存训练进度。
            wandb: Wandb对象，用于保存训练日志。
            save_dir: 检查点保存的目录，默认为'../checkpoints'，用于保存和加载检查点文件。
    Returns:    如果model参数为None且成功加载检查点，则返回加载的检查点数据字典；否则返回None。
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if vlm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{vlm_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        # 移除vision_encoder参数（不需要保存，因为是预训练的）
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def vlm_collate_fn(batch):
    """自定义的collate函数，用于将一个批次的数据样本组合成模型输入的格式。
    Args:    batch: 一个包含多个数据样本的列表，每个样本是一个元组，包含输入ID、标签和像素数据等信息。
    return: input_ids: 组合后的输入ID张量，形状为 (batch_size, seq_length)，用于模型的文本输入。
            labels: 组合后的标签张量，形状为 (batch_size, seq_length)，用于模型的训练目标。
            pixel_values: 组合后的像素数据张量，形状根据输入数据的格式而定，通常为 (batch_size, channels, height, width)，用于模型的图像输入。
    """
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
    
    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)

