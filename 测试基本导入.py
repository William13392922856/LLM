"""测试基本库导入"""
print("开始测试基本库导入...")

try:
    print("正在导入os...")
    import os
    print("✅ 导入os成功")
    
    print("正在导入torch...")
    import torch
    print(f"✅ 导入torch成功，版本: {torch.__version__}")
    
    print("正在导入numpy...")
    import numpy
    print(f"✅ 导入numpy成功，版本: {numpy.__version__}")
    
    print("\n✅ 测试完成！")
    
except Exception as e:
    print(f"❌ 测试过程中出错: {e}")
    import traceback
    traceback.print_exc()
