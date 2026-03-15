"""测试处理后数据集的结构"""
import json
import traceback
from pathlib import Path

print("开始测试数据集结构...")

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
    
    print("\n4. 测试分词和数据集创建...")
    def 预处理函数(样本):
        编码结果 = tokenizer(
            样本["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        return 编码结果
    
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": 训练文本})
    print(f"   原始数据集特征: {dataset.features}")
    
    # 测试map操作
    处理后数据集 = dataset.map(预处理函数, batched=False, remove_columns=["text"])
    print(f"   处理后数据集特征: {处理后数据集.features}")
    print(f"   处理后数据集大小: {len(处理后数据集)}")
    
    # 详细检查第一个样本
    print("\n5. 检查第一个样本...")
    第一个样本 = 处理后数据集[0]
    print(f"   第一个样本键: {list(第一个样本.keys())}")
    
    for 键 in 第一个样本:
        值 = 第一个样本[键]
        print(f"   {键}:")
        print(f"     类型: {type(值)}")
        print(f"     长度: {len(值) if hasattr(值, '__len__') else 'N/A'}")
        if isinstance(值, list) and 值:
            print(f"     第一个元素类型: {type(值[0])}")
            print(f"     第一个元素: {值[0]}")
            print(f"     最后一个元素: {值[-1]}")
    
    # 检查是否有text字段
    print("\n6. 检查是否有text字段...")
    if "text" in 处理后数据集.features:
        print("   ❌ 处理后数据集仍然包含text字段！")
    else:
        print("   ✅ 处理后数据集已移除text字段")
    
    # 检查所有样本的input_ids类型
    print("\n7. 检查所有样本的input_ids类型...")
    有问题的样本 = 0
    for i, 样本 in enumerate(处理后数据集):
        input_ids = 样本.get('input_ids')
        if not isinstance(input_ids, list):
            print(f"   ❌ 样本{i}的input_ids类型错误: {type(input_ids)}")
            有问题的样本 += 1
        elif not input_ids:
            print(f"   ❌ 样本{i}的input_ids为空")
            有问题的样本 += 1
        elif not isinstance(input_ids[0], int):
            print(f"   ❌ 样本{i}的input_ids元素类型错误: {type(input_ids[0])}")
            有问题的样本 += 1
    
    if 有问题的样本 == 0:
        print("   ✅ 所有样本的input_ids类型正确")
    else:
        print(f"   ❌ 发现{有问题的样本}个有问题的样本")
    
    print("\n✅ 数据集结构测试完成！")
    
except Exception as e:
    print(f"\n❌ 测试过程中出错: {e}")
    traceback.print_exc()
