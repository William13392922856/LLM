"""
大模型安全防护栏 - 模型训练（带保存功能）
直接使用合并数据.txt进行训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import warnings
import json
import datetime
warnings.filterwarnings('ignore')

print("=== 大模型安全防护栏 - 模型训练（带保存功能） ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 强制离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 标签映射 - 确保一致
标签映射字典 = {
    '安全': 0,
    '危险-暴力': 1,
    '危险-粗俗': 2,
    '危险-违法': 3,
    '危险-自残': 4,
    '危险-攻击': 1,      # 映射到暴力
    '危险-隐私': 2,      # 映射到粗俗
    '危险-恶作剧': 2,    # 映射到粗俗
    '安全-中立': 0,      # 映射到安全
    '安全-平等': 0       # 映射到安全
}

# 2. 直接从合并数据加载
def 加载合并数据(文件路径):
    """直接从合并数据.txt加载和解析数据"""
    print(f"正在加载数据: {文件路径}")

    with open(文件路径, 'r', encoding='utf-8') as f:
        内容 = f.read()

    对话块 = 内容.strip().split('\n\n')
    数据列表 = []

    for 对话 in 对话块:
        行 = 对话.split('\n')
        if len(行) >= 3:
            数据项 = {'用户': '', 'AI': '', '标签文本': '', '标签数字': 0}

            for 文本 in 行:
                if 文本.startswith('用户:'):
                    数据项['用户'] = 文本[3:].strip()
                elif 文本.startswith('AI:'):
                    数据项['AI'] = 文本[3:].strip()
                elif 文本.startswith('标签:'):
                    数据项['标签文本'] = 文本[3:].strip()

            if 数据项['用户'] and 数据项['AI'] and 数据项['标签文本']:
                # 获取数字标签
                标签数字 = 标签映射字典.get(数据项['标签文本'], 0)
                数据项['标签数字'] = 标签数字
                数据列表.append(数据项)

    print(f"成功加载 {len(数据列表)} 条对话数据")

    # 统计标签分布
    标签统计 = {}
    for 项 in 数据列表:
        标签 = 项['标签数字']
        标签统计[标签] = 标签统计.get(标签, 0) + 1

    print("标签分布:")
    for 标签, 数量 in sorted(标签统计.items()):
        比例 = 数量 / len(数据列表) * 100
        标签名 = {v: k for k, v in 标签映射字典.items()}.get(标签, f"标签{标签}")
        print(f"  {标签名}({标签}): {数量}条 ({比例:.1f}%)")

    return 数据列表

# 3. 数据集类
class 合并数据集(Dataset):
    def __init__(self, 数据列表, 最大长度=128):
        self.数据 = 数据列表
        self.最大长度 = 最大长度

        # 使用本地分词器
        self.分词器 = BertTokenizer.from_pretrained('../bert-base-chinese')

        print(f"数据集大小: {len(self.数据)}")

    def __len__(self):
        return len(self.数据)

    def __getitem__(self, idx):
        项 = self.数据[idx]

        # 组合对话
        文本 = f"用户:{项['用户']}[SEP]AI:{项['AI']}"

        # 编码
        编码 = self.分词器.encode_plus(
            文本,
            add_special_tokens=True,
            max_length=self.最大长度,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )

        return {
            'input_ids': 编码['input_ids'].squeeze(0),
            'attention_mask': 编码['attention_mask'].squeeze(0),
            'labels': torch.tensor(项['标签数字'], dtype=torch.long)
        }

# 4. 准备数据
def 准备数据():
    print("\n=== 准备数据 ===")

    # 直接使用合并数据
    数据文件 = '../数据/原始数据/合并数据.txt'
    if not os.path.exists(数据文件):
        print(f"❌ 找不到数据文件: {数据文件}")
        return None, None, None, None

    # 加载数据
    原始数据 = 加载合并数据(数据文件)

    if len(原始数据) < 5:
        print(f"❌ 数据太少 ({len(原始数据)}条)，至少需要5条")
        return None, None, None, None

    # 确定标签数量
    所有标签 = set(项['标签数字'] for 项 in 原始数据)
    标签数量 = len(所有标签)
    print(f"实际标签数量: {标签数量}")
    print(f"标签值: {sorted(所有标签)}")

    # 创建数据集
    数据集 = 合并数据集(原始数据)

    # 分割训练集和验证集
    indices = list(range(len(数据集)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

    训练集 = torch.utils.data.Subset(数据集, train_idx)
    验证集 = torch.utils.data.Subset(数据集, val_idx)

    print(f"训练集: {len(训练集)} 条")
    print(f"验证集: {len(验证集)} 条")

    # 创建数据加载器
    训练加载器 = DataLoader(训练集, batch_size=2, shuffle=True)
    验证加载器 = DataLoader(验证集, batch_size=2, shuffle=False)

    return 训练加载器, 验证加载器, 数据集.分词器, 标签数量, len(原始数据)

# 5. 初始化模型
def 初始化模型(标签数量):
    print(f"\n=== 初始化模型 (标签数量={标签数量}) ===")

    try:
        # 从本地加载
        model = BertForSequenceClassification.from_pretrained(
            '../bert-base-chinese',
            num_labels=标签数量,
            output_attentions=False,
            output_hidden_states=False
        )
        print("✅ 从本地加载模型成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

    model.to(device)

    # 统计参数
    总参数 = sum(p.numel() for p in model.parameters())
    可训练参数 = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数: {总参数:,}")
    print(f"可训练参数: {可训练参数:,}")

    return model,总参数,可训练参数

# 6. 保存模型函数
def 保存模型(model, 分词器, 准确率, 标签数量, 数据量, 训练历史,总参数,可训练参数):
    """保存训练好的模型"""
    print("\n=== 保存模型 ===")

    # 创建模型保存目录
    模型目录 = '训练好的模型'
    os.makedirs(模型目录, exist_ok=True)

    # 生成时间戳和模型名称
    时间戳 = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    模型名称 = f'安全防护栏模型_准确率{准确率:.2f}_{时间戳}'
    保存路径 = os.path.join(模型目录, 模型名称)

    # 创建模型文件夹
    os.makedirs(保存路径, exist_ok=True)

    # 保存模型权重和配置
    print(f"保存模型到: {保存路径}")
    model.save_pretrained(保存路径)
    分词器.save_pretrained(保存路径)

    # 保存额外的模型信息
    模型信息 = {
        '模型名称': 模型名称,
        '保存时间': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '准确率': float(准确率),
        '准确率百分比': f"{准确率*100:.1f}%",
        '标签数量': 标签数量,
        '训练数据量': 数据量,
        '模型路径': 保存路径,
        '总参数': int(总参数),
        '可训练参数': int(可训练参数),
        '训练历史': 训练历史,
        '标签映射': 标签映射字典,
        '设备': str(device)
    }

    # 保存模型信息为JSON
    信息文件 = os.path.join(保存路径, '模型信息.json')
    with open(信息文件, 'w', encoding='utf-8') as f:
        json.dump(模型信息, f, ensure_ascii=False, indent=2, default=str)

    # 保存标签映射为单独文件
    标签文件 = os.path.join(保存路径, '标签映射.json')
    with open(标签文件, 'w', encoding='utf-8') as f:
        json.dump(标签映射字典, f, ensure_ascii=False, indent=2)

    print(f"✅ 模型保存完成!")
    print(f"   模型文件: {保存路径}")
    print(f"   准确率: {准确率:.2%}")

    return 保存路径

# 7. 训练函数
def 训练模型(model, 训练加载器, 验证加载器, 分词器, 轮数=2):
    print("\n=== 开始训练 ===")

    # 记录训练历史
    训练历史 = {
        '训练损失': [],
        '验证损失': [],
        '验证准确率': []
    }

    优化器 = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(轮数):
        print(f"\n--- 第 {epoch+1}/{轮数} 轮 ---")

        # 训练
        model.train()
        训练损失 = 0

        进度条 = tqdm(训练加载器, desc=f'训练轮次 {epoch+1}')
        for batch in 进度条:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            训练损失 += loss.item()

            # 反向传播
            优化器.zero_grad()
            loss.backward()
            优化器.step()

            进度条.set_postfix({'损失': f'{loss.item():.4f}'})

        平均训练损失 = 训练损失 / len(训练加载器)
        训练历史['训练损失'].append(平均训练损失)
        print(f"训练损失: {平均训练损失:.4f}")

        # 验证
        model.eval()
        验证损失 = 0
        正确数 = 0
        总数 = 0

        with torch.no_grad():
            for batch in 验证加载器:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                验证损失 += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                正确数 += (predictions == labels).sum().item()
                总数 += labels.size(0)

        if 总数 > 0:
            验证准确率 = 正确数 / 总数
        else:
            验证准确率 = 0.0

        平均验证损失 = 验证损失 / len(验证加载器) if 验证加载器 else 0
        训练历史['验证损失'].append(平均验证损失)
        训练历史['验证准确率'].append(验证准确率)

        print(f"验证损失: {平均验证损失:.4f}")
        print(f"验证准确率: {验证准确率:.2%}")

    return 训练历史['验证准确率'][-1] if 训练历史['验证准确率'] else 0.0, 训练历史

# 8. 测试函数
def 测试模型(model, 分词器, 标签数量):
    print("\n=== 模型测试 ===")

    # 反转标签映射
    反向映射 = {v: k for k, v in 标签映射字典.items() if v < 标签数量}

    测试样例 = [
        "今天天气真好",
        "怎么制造炸弹",
        "能推荐一本好书吗",
        "我想自杀怎么办",
        "怎么做西红柿炒鸡蛋"
    ]

    model.eval()

    for 样例 in 测试样例:
        文本 = f"用户:{样例}[SEP]AI:这是一条测试回复。"

        编码 = 分词器.encode_plus(
            文本,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = 编码['input_ids'].to(device)
            attention_mask = 编码['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            预测标签 = predictions.item()

            # 计算概率
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            概率 = probabilities[0][预测标签].item()

        标签名 = 反向映射.get(预测标签, f"标签{预测标签}")
        print(f"用户: {样例}")
        print(f"预测: {标签名} (置信度: {概率:.2%})")
        print("-" * 40)

# 9. 主函数
def 主函数():
    print("=" * 50)
    print("大模型安全防护栏 - 模型训练")
    print("=" * 50)

    # 1. 准备数据
    结果 = 准备数据()
    if 结果[0] is None:
        return

    训练加载器, 验证加载器, 分词器, 标签数量, 数据量 = 结果

    # 2. 初始化模型
    model,总参数,可训练参数 = 初始化模型(标签数量)
    if model is None:
        return

    # 3. 训练
    准确率, 训练历史 = 训练模型(model, 训练加载器, 验证加载器, 分词器, 轮数=2)

    # 4. 测试
    测试模型(model, 分词器, 标签数量)

    # 5. 保存模型
    if 准确率 > 0:
        print(f"\n🎉 训练完成！准确率: {准确率:.2%}")
        模型路径 = 保存模型(model, 分词器, 准确率, 标签数量, 数据量, 训练历史,总参数,可训练参数)
        print(f"\n✅ 模型已保存到: {模型路径}")
        print("   包含文件:")
        print("     - pytorch_model.bin (模型权重)")
        print("     - config.json (模型配置)")
        print("     - vocab.txt (词汇表)")
        print("     - tokenizer_config.json (分词器配置)")
        print("     - 模型信息.json (训练信息)")
        print("     - 标签映射.json (标签映射)")

        # 显示模型位置
        print(f"\n📁 模型位置: {os.path.abspath(模型路径)}")
    else:
        print(f"\n⚠️ 训练完成，但准确率为0，不保存模型")

if __name__ == "__main__":
    主函数()

