"""检查原始数据集"""
import json

print("开始检查原始数据集...")

原始数据路径 = "数据\\原始数据\\推理数据集.json"

try:
    with open(原始数据路径, 'r', encoding='utf-8') as f:
        原始数据 = json.load(f)
    
    print(f"原始数据集样本数量: {len(原始数据)}")
    
    # 检查数据结构
    if 原始数据:
        print(f"第一个样本的键: {list(原始数据[0].keys())}")
        print(f"第一个样本内容: {原始数据[0]}")
    
    # 统计危险和安全样本
    if isinstance(原始数据, list):
        危险样本 = sum(1 for d in 原始数据 if d.get('标签') == 1)
        安全样本 = sum(1 for d in 原始数据 if d.get('标签') == 0)
        print(f"危险样本: {危险样本}")
        print(f"安全样本: {安全样本}")
        print(f"总计: {危险样本 + 安全样本}")
    
    print("\n✅ 检查完成！")
    
except Exception as e:
    print(f"❌ 检查过程中出错: {e}")
    import traceback
    traceback.print_exc()
