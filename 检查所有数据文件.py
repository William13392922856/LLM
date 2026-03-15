"""检查所有数据文件的样本数量"""
import json
import os
from pathlib import Path

print("开始检查所有数据文件...")

数据目录 = Path("数据/处理数据")
数据文件列表 = [
    "拆分数据集.json",
    "训练数据.json", 
    "训练集.json",
    "验证集.json",
    "测试集.json"
]

for 文件名 in 数据文件列表:
    文件路径 = 数据目录 / 文件名
    if 文件路径.exists():
        try:
            with open(文件路径, 'r', encoding='utf-8') as f:
                数据 = json.load(f)
            print(f"{文件名}: {len(数据)} 个样本")
            
            # 如果是训练集，显示详细信息
            if 文件名 == "训练集.json":
                危险样本 = sum(1 for d in 数据 if d.get('标签') == 1)
                安全样本 = sum(1 for d in 数据 if d.get('标签') == 0)
                print(f"  - 危险样本: {危险样本}")
                print(f"  - 安全样本: {安全样本}")
                
        except Exception as e:
            print(f"{文件名}: 读取失败 - {e}")
    else:
        print(f"{文件名}: 文件不存在")

print("\n✅ 检查完成！")
