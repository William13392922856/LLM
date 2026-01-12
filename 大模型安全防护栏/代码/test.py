# 测试脚本 test_all.py
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

print("=== 测试所有组件 ===")

# 1. 测试本地模型
print("1. 测试本地模型...")
try:
    tokenizer = BertTokenizer.from_pretrained('D:/PythonProject/LLM/bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('D:/PythonProject/LLM/bert-base-chinese')
    print("✅ 本地模型加载成功")
except Exception as e:
    print(f"❌ 本地模型加载失败: {e}")

# 2. 测试数据文件
print("\n2. 测试数据文件...")
数据文件 = '../数据/处理后的数据/对话数据_带数字标签.csv'
if os.path.exists(数据文件):
    df = pd.read_csv(数据文件)
    print(f"✅ 数据文件存在，行数: {len(df)}")
    print(f"   列名: {list(df.columns)}")
    print(f"   标签统计:")
    print(df['标签'].value_counts())
else:
    print("❌ 数据文件不存在")

# 3. 测试简单推理
print("\n3. 测试简单推理...")
try:
    inputs = tokenizer("今天天气真好", return_tensors="pt")
    outputs = model(**inputs)
    print(f"✅ 推理测试成功")
    print(f"   输出形状: {outputs.logits.shape}")
except Exception as e:
    print(f"❌ 推理失败: {e}")
