"""测试数据格式化和数据集创建（不加载分词器）"""
import json
import traceback
from pathlib import Path

print("开始测试数据处理...")

try:
    # 设置路径
    当前目录 = Path(__file__).parent
    训练数据路径 = 当前目录 / "数据" / "处理数据" / "训练集.json"
    
    print("\n1. 加载训练数据...")
    with open(训练数据路径, 'r', encoding='utf-8') as f:
        训练数据 = json.load(f)
    print(f"   ✅ 训练数据数量: {len(训练数据)}")
    
    print("\n2. 检查数据格式...")
    for i, 样本 in enumerate(训练数据[:3]):
        print(f"   样本{i+1}的键: {list(样本.keys())}")
        print(f"   提示词: {样本.get('提示词', '无')[:50]}...")
        print(f"   推理过程: {样本.get('推理过程', 样本.get(' 推理过程', '无'))[:50]}...")
    
    print("\n3. 格式化样本...")
    def 格式化样本(样本):
        提示词 = 样本.get('提示词', '')
        推理过程 = 样本.get('推理过程', '')
        if not 推理过程:
            推理过程 = 样本.get(' 推理过程', '')
        return f"提示词: {提示词}\n安全分析: {推理过程}"
    
    训练文本 = [格式化样本(样本) for 样本 in 训练数据]
    print(f"   ✅ 格式化后的文本数量: {len(训练文本)}")
    print(f"   第一个样本长度: {len(训练文本[0])}")
    
    print("\n4. 测试数据集创建...")
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": 训练文本})
    print(f"   ✅ 数据集创建成功")
    print(f"   数据集大小: {len(dataset)}")
    print(f"   数据集特征: {dataset.features}")
    
    print("\n✅ 数据处理测试完成！")
    
except Exception as e:
    print(f"\n❌ 测试过程中出错: {e}")
    traceback.print_exc()
