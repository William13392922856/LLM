"""
模型训练器 - 使用本地模型训练安全防护栏
严格按照配置文件中的参数进行训练
"""

import torch
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime
import os

class 模型训练器:
    def __init__(self):
        # 设置路径
        self.当前目录 = Path(__file__).parent.parent
        self.配置路径 = self.当前目录 / "配置文件" / "配置.yaml"
        self.训练数据路径 = self.当前目录 / "数据" / "处理数据" / "训练集.json"
        self.模型保存路径 = self.当前目录 / "模型文件"

        # ✅ 修复：先设置日志！
        self.设置日志()

        # ✅ 然后加载配置（此时self.日志已存在）
        self.配置 = self.加载配置()

        # 确保目录存在
        self.模型保存路径.mkdir(parents=True, exist_ok=True)

        # 验证配置
        if self.配置:
            self.验证配置()

    def 设置日志(self):
        """设置训练日志"""
        日志目录 = self.当前目录 / "日志文件"
        日志目录.mkdir(parents=True, exist_ok=True)

        日志文件 = 日志目录 / f"训练日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(日志文件, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.日志 = logging.getLogger(__name__)

    def 加载配置(self):
        """加载配置文件"""
        try:
            if not self.配置路径.exists():
                self.日志.error(f"找不到配置文件: {self.配置路径}")
                return {}

            with open(self.配置路径, 'r', encoding='utf-8') as f:
                配置 = yaml.safe_load(f)

            self.日志.info(f"✅ 配置文件加载成功")
            return 配置

        except Exception as e:
            self.日志.error(f"加载配置失败: {e}")
            return {}

    def 验证配置(self):
        """验证配置参数的完整性"""
        if not self.配置:
            self.日志.error("配置为空")
            return False

        # 检查关键配置
        必要配置 = {
            '模型': ['基础模型', 'LoRA参数r', 'LoRA参数alpha', 'LoRA层'],
            '训练': ['学习率', '训练轮数', '批次大小', '梯度累积', '热身步数', '日志间隔', '保存间隔']
        }

        for 模块, 参数列表 in 必要配置.items():
            if 模块 not in self.配置:
                self.日志.error(f"❌ 配置缺少模块: {模块}")
                return False

            for 参数 in 参数列表:
                if 参数 not in self.配置[模块]:
                    self.日志.warning(f"⚠️  {模块}.{参数} 未设置，使用默认值")

        return True

    def 开始训练(self):
        """开始模型训练流程"""
        print("="*60)
        print("🎯 开始模型训练 - 本地模型版本")
        print("="*60)

        # 1. 检查训练数据
        if not self.检查训练数据():
            return False

        # 2. 显示配置信息
        self.显示配置信息()

        # 3. 询问是否开始训练
        确认 = input("\n⚠️  是否开始训练? (y/N): ").strip().lower()
        if 确认 != 'y':
            print("训练已取消")
            return False

        # 4. 执行训练
        return self.执行训练()

    def 检查训练数据(self):
        """检查训练数据是否存在"""
        if not self.训练数据路径.exists():
            print("❌ 找不到训练数据")
            print(f"请先运行'处理数据'生成训练集")
            print(f"预期路径: {self.训练数据路径}")
            return False

        try:
            with open(self.训练数据路径, 'r', encoding='utf-8') as f:
                训练数据 = json.load(f)

            if not 训练数据:
                print("❌ 训练数据为空")
                return False

            print(f"✅ 找到训练数据: {len(训练数据)} 个样本")
            return True

        except Exception as e:
            print(f"❌ 加载训练数据失败: {e}")
            return False

    def 显示配置信息(self):
        """显示当前配置信息"""
        print("\n📊 当前配置:")
        print("="*40)

        # 模型配置
        print("🤖 模型配置:")
        if '模型' in self.配置:
            模型配置 = self.配置['模型']
            print(f"   基础模型: {模型配置.get('基础模型', '未设置')}")
            print(f"   LoRA参数r: {模型配置.get('LoRA参数r', '未设置')}")
            print(f"   LoRA参数alpha: {模型配置.get('LoRA参数alpha', '未设置')}")
            print(f"   dropout率: {模型配置.get('dropout率', '未设置')}")
            print(f"   LoRA层: {模型配置.get('LoRA层', '未设置')}")

        # 训练配置
        print("\n🎯 训练配置:")
        if '训练' in self.配置:
            训练配置 = self.配置['训练']
            print(f"   学习率: {训练配置.get('学习率', '未设置')}")
            print(f"   训练轮数: {训练配置.get('训练轮数', '未设置')}")
            print(f"   批次大小: {训练配置.get('批次大小', '未设置')}")
            print(f"   梯度累积: {训练配置.get('梯度累积', '未设置')}")
            print(f"   热身步数: {训练配置.get('热身步数', '未设置')}")
            print(f"   日志间隔: {训练配置.get('日志间隔', '未设置')}")
            print(f"   保存间隔: {训练配置.get('保存间隔', '未设置')}")

        # 数据统计
        try:
            with open(self.训练数据路径, 'r', encoding='utf-8') as f:
                训练数据 = json.load(f)

            危险样本 = sum(1 for 数据 in 训练数据 if 数据.get('标签') == 1)
            安全样本 = sum(1 for 数据 in 训练_data if 数据.get('标签') == 0)

            print("\n📈 数据统计:")
            print(f"   总样本数: {len(训练数据)}")
            print(f"   危险样本: {危险样本}")
            print(f"   安全样本: {安全样本}")
            if 训练数据:
                print(f"   危险比例: {危险样本/len(训练_data)*100:.1f}%")
        except:
            print("\n📈 数据统计: 无法加载数据")

        print("="*40)

    def 执行训练(self):
        """执行模型训练 - 严格使用本地模型"""
        print("\n" + "="*60)
        print("🚀 开始模型训练 (本地模型)")
        print("="*60)

        try:
            # 1. 获取本地模型路径
            model_config = self.配置.get('模型', {})
            基础模型配置 = model_config.get('基础模型', 'Qwen/Qwen2.5-0.5B-Instruct')

            print(f"📁 配置中的模型路径: {基础模型配置}")

            # 判断是否是本地路径
            本地模型路径 = None

            # 检查是否是本地路径
            if os.path.exists(基础模型配置):
                本地模型路径 = 基础模型配置
                print(f"✅ 找到本地模型 (绝对路径): {本地模型路径}")
            elif os.path.exists(os.path.join(self.当前目录, 基础模型配置)):
                本地模型路径 = os.path.join(self.当前目录, 基础模型配置)
                print(f"✅ 找到本地模型 (相对路径): {本地模型路径}")
            else:
                # 尝试在模型文件目录中查找
                模型目录 = self.当前目录 / "模型文件"
                if 模型目录.exists():
                    可能路径 = list(模型目录.rglob("*"))
                    if 可能路径:
                        for 路径 in 可能路径:
                            if 路径.is_dir() and any(路径.glob("*.safetensors")) or any(路径.glob("*.bin")):
                                本地模型路径 = str(路径)
                                print(f"✅ 在模型文件目录中找到: {本地模型路径}")
                                break

            if 本地模型路径 is None:
                print("❌ 找不到本地模型，请确保模型文件已下载到以下位置之一:")
                print(f"   1. {基础模型配置}")
                print(f"   2. {os.path.join(self.当前目录, 基础模型配置)}")
                print(f"   3. {self.当前目录 / '模型文件'}")
                return False

            print(f"✅ 使用本地模型: {本地模型路径}")

            # 2. 加载本地模型和分词器
            print("\n1️⃣ 加载本地模型和分词器...")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                # 强制只使用本地文件
                tokenizer = AutoTokenizer.from_pretrained(
                    本地模型路径,
                    local_files_only=True,
                    trust_remote_code=True
                )

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                print("   ✅ 分词器加载成功")

            except Exception as e:
                print(f"   ❌ 分词器加载失败: {e}")
                print("   尝试使用基础模型名称加载...")
                tokenizer = AutoTokenizer.from_pretrained(基础模型配置)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

            # 加载模型
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   使用设备: {device}")

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    本地模型路径,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    local_files_only=True,
                    trust_remote_code=True
                )
                print("   ✅ 模型加载成功")
            except Exception as e:
                print(f"   ❌ 模型加载失败: {e}")
                return False

            # 3. 配置LoRA
            print("\n2️⃣ 配置LoRA...")
            from peft import LoraConfig, get_peft_model, TaskType

            lora_r = model_config.get('LoRA参数r', 16)
            lora_alpha = model_config.get('LoRA参数alpha', 32)
            lora_dropout = model_config.get('dropout率', 0.1)
            target_modules = model_config.get('LoRA层', ["q_proj", "k_proj", "v_proj", "o_proj"])

            print(f"   LoRA参数 r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # 4. 准备训练数据
            print("\n3️⃣ 准备训练数据...")
            with open(self.训练数据路径, 'r', encoding='utf-8') as f:
                训练数据 = json.load(f)

            # 转换数据格式
            def 格式化样本(样本):
                提示词 = 样本.get('提示词', '')
                推理过程 = 样本.get('推理过程', '')
                return f"提示词: {提示词}\n安全分析: {推理过程}"

            训练文本 = [格式化样本(样本) for 样本 in 训练数据]

            # 创建数据集
            from datasets import Dataset
            dataset = Dataset.from_dict({"text": 训练文本})

            def 预处理函数(样本):
                # 使用 return_tensors="pt" 返回PyTorch张量
                编码结果 = tokenizer(
                    样本["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"  # ✅ 关键：返回张量格式
                )

                # 从批处理中提取单个样本
                return {key: val[0] for key, val in 编码结果.items()}

            处理后数据集 = dataset.map(预处理函数, batched=False)  # ✅ 单样本处理

            # 5. 设置训练参数（严格按照配置）
            print("\n4️⃣ 配置训练参数...")
            from transformers import TrainingArguments

            训练配置 = self.配置.get('训练', {})

            training_args = TrainingArguments(
                output_dir=str(self.模型保存路径 / "训练输出"),
                num_train_epochs=训练配置.get('训练轮数', 3),
                per_device_train_batch_size=训练配置.get('批次大小', 4),
                gradient_accumulation_steps=训练配置.get('梯度累积', 4),
                warmup_steps=训练配置.get('热身步数', 100),
                logging_steps=训练配置.get('日志间隔', 10),
                save_steps=训练配置.get('保存间隔', 100),
                learning_rate=训练配置.get('学习率', 0.0002),
                fp16=torch.cuda.is_available(),
                logging_dir=str(self.当前目录 / "日志文件" / "训练日志"),
                report_to="none",
                remove_unused_columns=False,
                save_total_limit=2,
                load_best_model_at_end=False,
                metric_for_best_model="loss",
                greater_is_better=False
            )

            # 6. 创建训练器
            from transformers import Trainer, DataCollatorForSeq2Seq

            data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=处理后数据集,
                data_collator=data_collator,
                tokenizer=tokenizer
            )

            # 7. 开始训练
            print("\n5️⃣ 开始训练...")
            print(f"   训练轮数: {训练配置.get('训练轮数', 3)}")
            print(f"   学习率: {训练配置.get('学习率', 0.0002)}")
            print(f"   批次大小: {训练配置.get('批次大小', 4)}")
            print(f"   梯度累积: {训练配置.get('梯度累积', 4)}")
            print(f"   总训练步数: {len(处理后数据集) // 训练配置.get('批次大小', 4) * 训练配置.get('训练轮数', 3)}")
            print("-"*40)

            trainer.train()

            # 8. 保存模型
            print("\n6️⃣ 保存模型...")
            模型保存名称 = f"安全防护栏模型_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            保存路径 = self.模型保存路径 / 模型保存名称
            保存路径.mkdir(parents=True, exist_ok=True)

            # 保存模型
            trainer.save_model(str(保存路径))
            tokenizer.save_pretrained(str(保存路径))

            # 保存模型信息
            模型信息 = {
                "模型名称": 模型保存名称,
                "基础模型": 基础模型配置,
                "本地路径": 本地模型路径,
                "训练时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "样本数量": len(训练数据),
                "训练参数": {
                    "学习率": 训练配置.get('学习率', 0.0002),
                    "训练轮数": 训练配置.get('训练轮数', 3),
                    "批次大小": 训练配置.get('批次大小', 4),
                    "梯度累积": 训练配置.get('梯度累积', 4),
                    "LoRA_rank": model_config.get('LoRA参数r', 16),
                    "LoRA_alpha": model_config.get('LoRA参数alpha', 32)
                },
                "模型文件": []
            }

            # 添加实际存在的文件
            for 文件 in 保存路径.glob("*"):
                模型信息["模型文件"].append(文件.name)

            模型信息文件 = 保存路径 / "模型信息.json"
            with open(模型信息文件, 'w', encoding='utf-8') as f:
                json.dump(模型信息, f, ensure_ascii=False, indent=2)

            # 更新主模型信息文件
            with open(self.模型保存路径 / "模型信息.json", 'w', encoding='utf-8') as f:
                json.dump(模型信息, f, ensure_ascii=False, indent=2)

            print("\n" + "="*60)
            print("✅ 模型训练完成！")
            print("="*60)
            print(f"\n📁 模型已保存到: {保存路径}")
            print("📄 模型文件列表:")
            for 文件 in 保存_path.glob("*"):
                print(f"  - {文件.name}")

            return True

        except ImportError as e:
            print(f"\n❌ 缺少必要的依赖库: {e}")
            print("请安装依赖: pip install transformers peft datasets")
            return False
        except Exception as e:
            print(f"\n❌ 训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False

# 测试代码
if __name__ == "__main__":
    print("🧪 测试模型训练器 (本地模型版本)...")

    try:
        训练器 = 模型训练器()

        # 显示配置
        训练器.显示配置信息()

        # 询问是否测试训练
        测试 = input("\n是否测试训练流程? (y/N): ").strip().lower()
        if 测试 == 'y':
            训练器.开始训练()
        else:
            print("测试完成，未开始训练")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()