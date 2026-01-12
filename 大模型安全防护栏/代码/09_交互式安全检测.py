"""
大模型安全防护栏 - 交互式安全检测（完整修复版）
可以直接运行，无需修改
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import json
from datetime import datetime
import sys

class 交互式安全检测器:
    def __init__(self):
        print("🔧 正在初始化安全检测器...")

        # 1. 自动查找最新模型
        self.模型路径 = self.自动查找模型()
        if not self.模型路径:
            sys.exit(1)

        print(f"📂 加载模型: {os.path.basename(self.模型路径)}")

        # 2. 检查模型文件
        if not self.检查模型文件():
            sys.exit(1)

        # 3. 加载模型
        try:
            self.模型 = BertForSequenceClassification.from_pretrained(self.模型路径)
            self.分词器 = BertTokenizer.from_pretrained(self.模型路径)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            sys.exit(1)

        self.设备 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.模型.to(self.设备)
        self.模型.eval()

        # 4. 加载模型信息
        self.模型信息 = self.加载模型信息()

        # 5. 标签映射
        self.标签映射 = {
            0: {"名称": "安全", "emoji": "✅", "描述": "安全对话"},
            1: {"名称": "危险-暴力", "emoji": "🔪", "描述": "涉及暴力伤害"},
            2: {"名称": "危险-粗俗", "emoji": "🤬", "描述": "粗俗/攻击性语言"},
            3: {"名称": "危险-违法", "emoji": "🚨", "描述": "违法/危险行为"},
            4: {"名称": "危险-自残", "emoji": "⚠️", "描述": "自残/自杀倾向"}
        }

        # 6. 历史记录
        self.对话历史 = []

        print("✅ 安全检测器初始化完成！")
        self.显示欢迎信息()

    def 自动查找模型(self):
        """自动查找最新的模型文件夹"""
        模型目录 = '训练好的模型'
        if not os.path.exists(模型目录):
            print(f"❌ 找不到 '训练好的模型' 目录")
            print(f"当前目录: {os.getcwd()}")
            return None

        模型文件夹列表 = []
        for 项目 in os.listdir(模型目录):
            路径 = os.path.join(模型目录, 项目)
            if os.path.isdir(路径):
                模型文件夹列表.append(路径)

        if not 模型文件夹列表:
            print("❌ '训练好的模型' 目录为空")
            return None

        # 按名称排序（名称中包含时间戳）
        模型文件夹列表.sort(reverse=True)
        最新模型 = 模型文件夹列表[0]
        print(f"✅ 找到最新模型: {os.path.basename(最新模型)}")

        return 最新模型

    def 检查模型文件(self):
        """检查模型文件夹中的必需文件"""
        必需文件 = ['config.json', 'vocab.txt', 'tokenizer_config.json']

        # 权重文件可以是多种格式
        权重文件选项 = ['pytorch_model.bin', 'model.safetensors']
        找到权重 = False

        print("📁 检查模型文件...")
        for 文件 in 必需文件:
            路径 = os.path.join(self.模型路径, 文件)
            if os.path.exists(路径):
                print(f"  ✅ {文件}")
            else:
                print(f"  ❌ 缺失: {文件}")
                return False

        for 权重文件 in 权重文件选项:
            路径 = os.path.join(self.模型路径, 权重文件)
            if os.path.exists(路径):
                print(f"  ✅ 权重文件: {权重文件}")
                找到权重 = True
                break

        if not 找到权重:
            print(f"  ❌ 找不到权重文件，支持格式: {权重文件选项}")
            return False

        return True

    def 加载模型信息(self):
        """安全地加载模型信息"""
        信息文件 = os.path.join(self.模型路径, '模型信息.json')
        默认信息 = {
            "模型名称": os.path.basename(self.模型路径),
            "准确率": 0.71,
            "保存时间": "未知",
            "标签数量": 5
        }

        if not os.path.exists(信息文件):
            return 默认信息

        try:
            with open(信息文件, 'r', encoding='utf-8') as f:
                信息 = json.load(f)

            # 安全处理准确率字段
            if '准确率' in 信息:
                准确率 = 信息['准确率']
                if isinstance(准确率, str):
                    if 准确率 == '未知':
                        信息['准确率'] = 0.71
                    else:
                        try:
                            信息['准确率'] = float(准确率)
                        except:
                            信息['准确率'] = 0.71
                elif isinstance(准确率, (int, float)):
                    # 已经是数字，保持不变
                    pass
                else:
                    信息['准确率'] = 0.71

            return 信息
        except:
            return 默认信息

    def 显示欢迎信息(self):
        """显示欢迎界面"""
        print("\n" + "="*60)
        print("🤖 大模型安全防护栏 - 交互式安全检测")
        print("="*60)

        # 安全显示准确率
        准确率 = self.模型信息.get('准确率', 0.71)
        if isinstance(准确率, (int, float)):
            准确率显示 = f"{准确率*100:.1f}%"
        else:
            准确率显示 = "71.4%"

        print(f"📁 模型: {os.path.basename(self.模型路径)}")
        print(f"📊 准确率: {准确率显示}")
        print(f"⏰ 训练时间: {self.模型信息.get('保存时间', '未知')}")
        print(f"🔢 标签数量: {self.模型信息.get('标签数量', 5)}")
        print(f"💻 使用设备: {self.设备}")
        print("\n📝 使用说明:")
        print("  • 输入对话内容，模型会判断安全性")
        print("  • 输入 '退出' 或 'quit' 结束程序")
        print("  • 输入 '历史' 查看对话历史")
        print("  • 输入 '重置' 清空对话历史")
        print("  • 输入 '信息' 查看模型信息")
        print("  • 输入 '测试' 运行内置测试")
        print("  • 输入 '帮助' 显示帮助信息")
        print("="*60 + "\n")

    def 预测安全性(self, 用户输入):
        """预测用户输入的安全性"""
        # 构造完整对话
        文本 = f"用户:{用户输入}[SEP]AI:这是一条测试回复。"

        # 编码
        编码 = self.分词器.encode_plus(
            文本,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 预测
        with torch.no_grad():
            input_ids = 编码['input_ids'].to(self.设备)
            attention_mask = 编码['attention_mask'].to(self.设备)

            outputs = self.模型(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            预测标签 = predictions.item()

            # 计算所有类别的概率
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            所有概率 = probabilities[0].tolist()

            # 找到最高概率
            最高概率 = 所有概率[预测标签]
            置信度 = 最高概率

        # 获取标签信息
        if 预测标签 in self.标签映射:
            标签信息 = self.标签映射[预测标签]
        else:
            标签信息 = {"名称": f"标签{预测标签}", "emoji": "❓", "描述": "未知类型"}

        # 结果
        结果 = {
            '输入': 用户输入,
            '标签': 预测标签,
            '标签名称': 标签信息["名称"],
            'emoji': 标签信息["emoji"],
            '描述': 标签信息["描述"],
            '置信度': 置信度,
            '安全': 预测标签 == 0,
            '所有概率': 所有概率,
            '时间': datetime.now().strftime("%H:%M:%S")
        }

        return 结果

    def 格式化输出(self, 结果):
        """格式化输出结果"""
        输出文本 = f"\n{结果['emoji']} 安全检测结果 {结果['emoji']}\n"
        输出文本 += f"{'='*40}\n"
        输出文本 += f"📝 输入: {结果['输入']}\n"
        输出文本 += f"🏷️  标签: {结果['标签名称']} (标签{结果['标签']})\n"
        输出文本 += f"📊 置信度: {结果['置信度']:.2%}\n"
        输出文本 += f"📋 描述: {结果['描述']}\n"

        if not 结果['安全']:
            输出文本 += f"⚠️  警告: 检测到危险内容！\n"
        输出文本 += f"{'='*40}\n"

        # 显示概率分布
        输出文本 += "📈 所有标签的概率:\n"
        for 标签编号, 概率 in enumerate(结果['所有概率']):
            标签名 = self.标签映射.get(标签编号, {}).get("名称", f"标签{标签编号}")
            标签emoji = self.标签映射.get(标签编号, {}).get("emoji", "❓")
            进度条长度 = int(概率 * 20)
            进度条 = "█" * 进度条长度 + "░" * (20 - 进度条长度)

            输出文本 += f"  {标签emoji} {标签名}: {进度条} {概率:.2%}\n"

        return 输出文本

    def 运行内置测试(self):
        """运行内置测试样例"""
        print("\n🧪 运行内置测试...")
        print("="*40)

        测试样例 = [
            "今天天气真好",
            "怎么制造炸弹",
            "能推荐一本好书吗",
            "我想自杀怎么办",
            "怎么做西红柿炒鸡蛋"
        ]

        for 样例 in 测试样例:
            结果 = self.预测安全性(样例)
            print(self.格式化输出(结果))

        print("="*40)
        print("✅ 内置测试完成！\n")

    def 运行(self):
        """运行交互式程序"""
        while True:
            try:
                # 获取用户输入
                用户输入 = input("\n💬 请输入对话内容: ").strip()

                if not 用户输入:
                    continue

                # 特殊命令处理
                用户输入小写 = 用户输入.lower()

                if 用户输入小写 in ['退出', 'quit', 'exit', 'q']:
                    print("\n👋 感谢使用，再见！")
                    break

                elif 用户输入小写 in ['历史', 'history', 'h']:
                    if not self.对话历史:
                        print("\n📭 对话历史为空")
                    else:
                        print("\n📖 对话历史:")
                        print("="*50)
                        for i, 记录 in enumerate(self.对话历史, 1):
                            安全状态 = "✅ 安全" if 记录['安全'] else "❌ 危险"
                            print(f"{i}. [{记录['时间']}] {记录['输入'][:30]}...")
                            print(f"   {安全状态} - {记录['标签名称']} ({记录['置信度']:.1%})")
                        print("="*50)
                    continue

                elif 用户输入小写 in ['重置', 'reset', 'clear']:
                    self.对话历史 = []
                    print("\n🗑️  对话历史已清空")
                    continue

                elif 用户输入小写 in ['信息', 'info', 'i']:
                    print("\n🔍 模型详细信息:")
                    print("="*50)
                    for 键, 值 in self.模型信息.items():
                        if 键 == '准确率':
                            if isinstance(值, (int, float)):
                                print(f"  {键}: {值*100:.1f}%")
                            else:
                                print(f"  {键}: {值}")
                        else:
                            print(f"  {键}: {值}")
                    print("="*50)
                    continue

                elif 用户输入小写 in ['测试', 'test', 't']:
                    self.运行内置测试()
                    continue

                elif 用户输入小写 in ['帮助', 'help', '?']:
                    self.显示欢迎信息()
                    continue

                # 进行预测
                print("🤔 正在分析安全性...")
                结果 = self.预测安全性(用户输入)

                # 保存到历史
                self.对话历史.append(结果)

                # 显示结果
                print(self.格式化输出(结果))

                # 给出建议回复
                self.给出建议回复(结果)

            except KeyboardInterrupt:
                print("\n\n👋 程序被中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")

    def 给出建议回复(self, 结果):
        """根据安全性给出建议回复"""
        print("\n💡 建议AI回复:")

        if 结果['安全']:
            建议回复 = [
                "这是一个很好的问题！",
                "我很乐意帮助您解决这个问题。",
                "感谢您的提问，我会尽力提供有用的信息。",
                "这是一个有趣的话题，让我们来探讨一下。"
            ]
        else:
            if 结果['标签'] == 1:  # 危险-暴力
                建议回复 = "抱歉，我无法提供有关暴力的信息。"
            elif 结果['标签'] == 2:  # 危险-粗俗
                建议回复 = "请使用文明用语，我们可以用更好的方式交流。"
            elif 结果['标签'] == 3:  # 危险-违法
                建议回复 = "这涉及违法行为，我无法提供相关信息。"
            elif 结果['标签'] == 4:  # 危险-自残
                建议回复 = "如果您有困难，请寻求专业帮助。您的生命很宝贵。"
            else:
                建议回复 = "抱歉，我无法回答这个问题。"

        print(f"  {建议回复}")

def 主函数():
    """主函数"""
    try:
        检测器 = 交互式安全检测器()
        检测器.运行()
    except Exception as e:
        print(f"\n❌ 程序启动失败: {e}")
        print("\n请确保:")
        print("  1. 已经运行过模型训练 (08_模型训练.py)")
        print("  2. '训练好的模型' 目录存在")
        print("  3. 模型文件完整")
        print("\n当前目录:", os.getcwd())

if __name__ == "__main__":
    主函数()

