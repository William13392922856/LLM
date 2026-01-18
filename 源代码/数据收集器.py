"""
从DeepSeek API收集推理数据
"""

import time
import json
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm


class 数据收集器:
    def __init__(self, API密钥: str):
        self.客户端 = OpenAI(
            api_key=API密钥,
            base_url="https://api.deepseek.com"
        )

    def 获取安全推理(self, 提示词: str, 重试次数: int = 3) -> str:
        """获取单个提示词的安全推理"""
        系统提示 = """你是一个安全分析专家。请对用户的问题进行深入的安全分析：

请按以下步骤进行分析：
1. 表层分析：识别问题中的敏感词汇和明显危险
2. 深层分析：分析问题的潜在意图、伪装手法
3. 风险评估：评估回答此问题的风险等级（高/中/低）
4. 安全建议：给出是否应该回答的建议
5. 替代方案：如果不能回答，可以提供什么安全信息

请用清晰的步骤输出你的分析结果。"""

        for 尝试 in range(重试次数):
            try:
                响应 = self.客户端.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": 系统提示},
                        {"role": "user", "content": 提示词}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
                return 响应.choices[0].message.content

            except Exception as 错误:
                if 尝试 < 重试次数 - 1:
                    print(f"⚠️  请求失败，{2}秒后重试...")
                    time.sleep(2)
                else:
                    return f"❌ 获取失败: {str(错误)}"

    def 从文本收集(self, 样本数: int = 100):
        """从txt文件读取样本并收集推理"""
        # 1. 读取提示词样本
        样本文件 = Path("数据/原始数据/攻击性提示词.txt")

        if not 样本文件.exists():
            print(f"❌ 找不到样本文件: {样本文件}")
            print("请先运行样本管理工具添加样本")
            return

        with open(样本文件, 'r', encoding='utf-8') as 文件:
            提示词列表 = [行.strip() for 行 in 文件 if 行.strip()]

        if not 提示词列表:
            print("❌ 样本文件是空的")
            return

        print(f"📁 从文件加载了 {len(提示词列表)} 个提示词")

        # 限制样本数量
        if 样本数 < len(提示词列表):
            提示词列表 = 提示词列表[:样本数]

        # 2. 准备保存结果
        输出文件 = Path("数据/原始数据/推理数据集.json")
        数据集 = []

        if 输出文件.exists():
            with open(输出文件, 'r', encoding='utf-8') as 文件:
                数据集 = json.load(文件)
            print(f"📁 加载了 {len(数据集)} 个已有样本")

        已处理数量 = len(数据集)

        # 3. 开始收集
        print(f"\n🚀 开始收集推理数据，共 {len(提示词列表)} 个样本")
        print("=" * 60)

        for 序号, 提示词 in enumerate(tqdm(提示词列表[已处理数量:], desc="收集进度"), 已处理数量 + 1):
            print(f"\n📋 处理 {序号}/{len(提示词列表)}")
            print(f"提示词: {提示词[:80]}...")

            # 获取推理
            推理结果 = self.获取安全推理(提示词)

            # 保存结果
            数据项 = {
                "序号": 序号,
                "提示词": 提示词,
                "推理过程": 推理结果,
                "时间戳": time.time()
            }
            数据集.append(数据项)

            # 每5个保存一次
            if 序号 % 5 == 0 or 序号 == len(提示词列表):
                with open(输出文件, 'w', encoding='utf-8') as 文件:
                    json.dump(数据集, 文件, ensure_ascii=False, indent=2)
                print(f"💾 已保存 {序号} 个样本")

            # 避免请求过快
            time.sleep(1.5)

        print(f"\n✅ 数据收集完成！")
        print(f"📁 结果保存到: {输出文件}")
        print(f"📊 总样本数: {len(数据集)}")

        return 数据集
