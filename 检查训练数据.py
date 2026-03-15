"""检查训练数据文件"""
import json
import traceback

print("开始检查训练数据文件...")

try:
    训练数据路径 = "数据\\处理数据\\训练集.json"
    
    print(f"检查文件路径: {训练数据路径}")
    
    # 检查文件是否存在
    from pathlib import Path
    path = Path(训练数据路径)
    print(f"文件存在: {path.exists()}")
    print(f"文件大小: {path.stat().st_size} 字节")
    
    # 读取文件
    with open(训练数据路径, 'r', encoding='utf-8') as f:
        训练数据 = json.load(f)
    
    print(f"数据类型: {type(训练数据)}")
    print(f"数据长度: {len(训练数据)}")
    
    if 训练数据:
        print(f"第一个样本的键: {list(训练数据[0].keys())}")
        print(f"第一个样本内容: {训练数据[0]}")
    
    print("\n✅ 训练数据检查完成！")
    
except Exception as e:
    print(f"❌ 检查过程中出错: {e}")
    traceback.print_exc()
