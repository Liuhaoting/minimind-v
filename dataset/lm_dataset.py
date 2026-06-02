import sys
import os
__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import torch
import io # 提供内存中的二进制流操作，用于从字节数据打开图片。
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model.model_vlm import MiniMindVLM
import pyarrow as pa # Apache Arrow 的 Python 接口，用于高效处理列式数据。 
import pyarrow.parquet as pq # Parquet 文件格式的读写库

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class VLMDataset(Dataset):
    """用于视觉语言模型训练的数据集类，继承自PyTorch的Dataset基类。
    该类负责从Parquet文件中加载对话数据和图像数据，并将它们预处理成适合模型输入的格式。
    Args:    parquet_path: Parquet文件的路径，包含对话和图像数据。
            tokenizer: 用于文本数据的分词器对象，通常是从预训练语言模型中获取的。
            preprocess: 用于图像数据预处理的函数或对象，通常是从预训练视觉模型中获取的。
            max_length: 输入文本的最大长度，超过该长度的文本将被截断。
            image_special_token: 用于替换对话中图像占位符的特殊标记字符串，默认为'<|image_pad|>'。
            image_token_len: 替换图像占位符的特殊标记的重复次数，默认为64，以确保足够的长度来表示图像信息。
    """
    def __init__(self, parquet_path, tokenizer, preprocess=None, max_length=512, image_special_token='<|image_pad|>', image_token_len=64):
        super().__init__()
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            content = turn['content'].replace('<image>', self.image_special_token) if turn.get('role') != 'system' else turn['content']
            messages.append({"role": turn['role'], "content": content})
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """根据输入的token ID序列生成对应的标签序列，用于语言模型的训练。标签序列中的值为-100表示该位置不参与损失计算，其他位置的值与输入ID相同。
        Args:    input_ids (List[int]): 输入的token ID列表，每个ID对应一个token，序列长度不超过max_length。
        Returns: List[int]: 生成的标签列表，与输入ID列表长度相同，其中包含了训练目标的token ID和-100的掩码值。"""
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        if not isinstance(image_bytes, list): image_bytes = [image_bytes]
        
        conversations = pre_processing_chat(conversations) # 在生成提示词之前添加系统提示词
        prompt = self.create_chat_prompt(conversations) # 生成提示词
        prompt = post_processing_chat(prompt) # 在生成提示词之后移除空思考标签
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length] # 将生成的提示词转换为token ID，并截断到最大长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) # 使用pad_token_id填充输入ID列表，使其长度达到max_length
        labels = self.generate_labels(input_ids) # 根据输入ID列表生成对应的标签列表，用于训练目标

        image_inputs_list = [MiniMindVLM.image2tensor(Image.open(io.BytesIO(img)), self.preprocess) for img in image_bytes]
        if hasattr(image_inputs_list[0], 'keys'):
            image_data = {k: torch.cat([inp[k] for inp in image_inputs_list], dim=0) for k in image_inputs_list[0].keys()}
        else:
            image_data = torch.stack(image_inputs_list)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), image_data

# 测试parquet数据读取和可视化
if __name__ == '__main__':
    import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    for path in ['pretrain_i2t.parquet', 'sft_i2t.parquet']:
        pf = pq.ParquetFile(path); n = pf.num_row_groups; t = pa.concat_tables([pf.read_row_group(i * n // 5).slice(0, 1) for i in range(5)]); fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i in range(5):
            img_data = t['image_bytes'][i].as_py(); img_data = img_data[0] if isinstance(img_data, list) else img_data
            ax[i].imshow(Image.open(io.BytesIO(img_data))); ax[i].axis('off')
            ax[i].set_title(json.loads(t['conversations'][i].as_py())[1]['content'][:30], fontsize=8)
        out = path.replace('.parquet', '_preview.png'); plt.savefig(out); print(f'已保存{out}, 共{pf.metadata.num_rows}条')
