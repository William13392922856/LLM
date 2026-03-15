"""测试导入transformers库"""
print("开始测试导入transformers...")

try:
    print("正在导入transformers...")
    import transformers
    print(f"✅ 导入transformers成功，版本: {transformers.__version__}")
    
    print("正在导入AutoTokenizer和AutoModelForCausalLM...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ 导入成功")
    
    print("\n✅ 测试完成！")
    
except Exception as e:
    print(f"❌ 测试过程中出错: {e}")
    import traceback
    traceback.print_exc()
