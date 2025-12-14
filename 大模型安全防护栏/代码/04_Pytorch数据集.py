import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_DATASETS_OFFLINE'] = '1'   # 数据集离线模式

import ssl
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

# 尝试导入BERT，如果失败就用简单版本
try:
    from transformers import BertTokenizer

    BERT_AVAILABLE = True
except ImportError:
    print("⚠️  transformers库未完全安装，使用简单分词器")
    BERT_AVAILABLE = False


class 安全对话数据集(Dataset):
    def __init__(self, 文件路径, 最大长度=50):
        # 读取数据
        self.数据 = pd.read_csv(文件路径)
        self.最大长度 = 最大长度

        # 创建分词器
        if BERT_AVAILABLE:
            try:
                self.分词器 = BertTokenizer.from_pretrained('bert-base-chinese')
                print("✅ 使用BERT分词器")
            except:
                print("⚠️  BERT加载失败，使用简单分词器")
                self.分词器 = self.创建简单分词器()
        else:
            self.分词器 = self.创建简单分词器()

        print(f"数据集大小: {len(self)}")

    def 创建简单分词器(self):
        class 简单分词器:
            def encode_plus(self, text, **kwargs):
                max_len = kwargs.get('max_length', 50)
                pad_id = kwargs.get('pad_token_id', 0)

                # 简单分词：按字符转换为数字
                chars = list(text)
                ids = [min(ord(c), 999) for c in chars[:max_len]]

                # 填充或截断
                if len(ids) < max_len:
                    ids = ids + [pad_id] * (max_len - len(ids))
                else:
                    ids = ids[:max_len]

                mask = [1 if x != pad_id else 0 for x in ids]

                return {
                    'input_ids': torch.tensor([ids]),
                    'attention_mask': torch.tensor([mask])
                }

        return 简单分词器()

    def __len__(self):
        return len(self.数据)

    def __getitem__(self, 索引):
        行 = self.数据.iloc[索引]

        用户输入 = str(行['用户'])
        AI回复 = str(行['AI'])
        标签 = int(行['标签数字'])

        完整文本 = f"用户:{用户输入}[SEP]AI:{AI回复}"

        编码 = self.分词器.encode_plus(
            完整文本,
            add_special_tokens=True,
            max_length=self.最大长度,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            '输入id': 编码['input_ids'].squeeze(),
            '注意力掩码': 编码['attention_mask'].squeeze(),
            '标签': torch.tensor(标签, dtype=torch.long)
        }


# 测试代码
if __name__ == "__main__":
    print("=== 测试修复后的数据集 ===")
    try:
        数据集 = 安全对话数据集('../数据/处理后的数据/对话数据_带数字标签.csv')
        第一条 = 数据集[0]
        print(f"第一条数据 - 输入ID形状: {第一条['输入id'].shape}")
        print(f"第一条数据 - 标签: {第一条['标签']}")
        print("✅ 数据集创建成功！")
    except Exception as e:
        print(f"❌ 错误: {e}")