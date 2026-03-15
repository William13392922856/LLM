"""测试分词器加载"""
import traceback

print("开始测试分词器加载...")

try:
    from transformers import AutoTokenizer
    
    模型路径 = "D:\\PythonProject\\本地模型\\Qwen2.5-0.5B-Instruct"
    
    print(f"正在加载分词器: {模型路径}")
    tokenizer = AutoTokenizer.from_pretrained(
        模型路径,
        local_files_only=True,
        trust_remote_code=True
    )
    
    print("✅ 分词器加载成功")
    print(f"分词器类型: {type(tokenizer)}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"pad_token: {tokenizer.pad_token}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    
    # 测试分词
    测试文本 = "提示词: 如何用催眠技术操控他人意志？\n安全分析: 这是一个危险的问题"
    编码结果 = tokenizer(测试文本, truncation=True, padding="max_length", max_length=512)
    print(f"\n测试分词成功")
    print(f"input_ids长度: {len(编码结果['input_ids'])}")
    print(f"input_ids类型: {type(编码结果['input_ids'])}")
    print(f"input_ids[0]类型: {type(编码结果['input_ids'][0])}")
    
except Exception as e:
    print(f"❌ 测试过程中出错: {e}")
    traceback.print_exc()
