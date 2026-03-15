"""测试数据处理和分词部分"""
import json
import traceback
from pathlib import Path

print("开始测试数据处理和分词...")

try:
    # 设置路径
    当前目录 = Path(__file__).parent
    训练数据路径 = 当前目录 / "数据" / "处理数据" / "训练集.json"
    模型路径 = "D:\\PythonProject\\本地模型\\Qwen2.5-0.5B-Instruct"
    
    print("\n1. 加载训练数据...")
    with open(训练数据路径, 'r', encoding='utf-8') as f:
        训练数据 = json.load(f)
    print(f"   ✅ 训练数据数量: {len(训练数据)}")
    
    print("\n2. 格式化样本...")
    def 格式化样本(样本):
        提示词 = 样本.get('提示词', '')
        推理过程 = 样本.get('推理过程', '')
        if not 推理过程:
            推理过程 = 样本.get(' 推理过程', '')
        return f"提示词: {提示词}\n安全分析: {推理过程}"
    
    训练文本 = [格式化样本(样本) for 样本 in 训练数据]
    print(f"   ✅ 格式化后的文本数量: {len(训练文本)}")
    print(f"   第一个样本长度: {len(训练文本[0])}")
    
    print("\n3. 加载分词器...")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        模型路径,
        local_files_only=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ 分词器加载成功")
    
    print("\n4. 测试分词...")
    def 预处理函数(样本):
        编码结果 = tokenizer(
            样本["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        return 编码结果
    
    # 测试前3个样本
    for i in range(min(3, len(训练文本))):
        测试样本 = {"text": 训练文本[i]}
        编码结果 = 预处理函数(测试样本)
        print(f"   样本{i+1}编码成功")
        print(f"   input_ids长度: {len(编码结果['input_ids'])}")
        print(f"   input_ids类型: {type(编码结果['input_ids'])}")
        print(f"   input_ids[0]类型: {type(编码结果['input_ids'][0])}")
    
    print("\n5. 测试数据集创建...")
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": 训练文本})
    处理后数据集 = dataset.map(预处理函数, batched=False)
    print(f"   ✅ 数据集创建成功")
    print(f"   处理后数据集大小: {len(处理后数据集)}")
    print(f"   处理后数据集特征: {处理后数据集.features}")
    
    # 检查第一个样本的特征
    第一个样本 = 处理后数据集[0]
    print(f"   第一个样本键: {list(第一个样本.keys())}")
    print(f"   input_ids长度: {len(第一个样本['input_ids'])}")
    print(f"   attention_mask长度: {len(第一个样本['attention_mask'])}")
    
    print("\n✅ 数据处理测试完成！")
    
except Exception as e:
    print(f"\n❌ 测试过程中出错: {e}")
    traceback.print_exc()
