"""最小化测试模型加载"""
import time
import torch

print("开始测试模型加载...")
print(f"PyTorch版本: {torch.__version__}")
print(f"可用设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")

start_time = time.time()

try:
    print("正在导入transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✅ 导入transformers成功")
    
    模型路径 = "D:\\PythonProject\\本地模型\\Qwen2.5-0.5B-Instruct"
    print(f"正在加载分词器: {模型路径}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        模型路径,
        local_files_only=True,
        trust_remote_code=True
    )
    print("✅ 分词器加载成功")
    print(f"分词器词汇表大小: {tokenizer.vocab_size}")
    
    # 加载模型
    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        模型路径,
        torch_dtype=torch.float16,
        device_map=None,
        local_files_only=True,
        trust_remote_code=True
    )
    print("✅ 模型加载成功")
    print(f"模型类型: {type(model)}")
    
    total_time = time.time() - start_time
    print(f"\n✅ 测试完成！总耗时: {total_time:.2f}秒")
    
except Exception as e:
    print(f"❌ 测试过程中出错: {e}")
    import traceback
    traceback.print_exc()
