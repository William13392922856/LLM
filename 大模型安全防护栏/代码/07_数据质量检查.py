"""
数据质量检查工具
检查生成的数据是否有问题
"""
import pandas as pd
import re
import os


def 检查数据质量(文件路径: str):
    """检查数据质量"""
    print(f"=== 检查数据质量: {文件路径} ===")

    with open(文件路径, 'r', encoding='utf-8') as f:
        内容 = f.read()

    # 分割对话
    对话块 = 内容.strip().split('\n\n')
    print(f"总对话数: {len(对话块)}")

    问题列表 = []

    for i, 对话 in enumerate(对话块):
        行列表 = 对话.split('\n')

        # 检查格式
        if len(行列表) != 3:
            问题列表.append(f"对话 {i + 1}: 行数不是3行")
            continue

        用户行, AI行, 标签行 = 行列表

        # 检查前缀
        if not 用户行.startswith('用户:'):
            问题列表.append(f"对话 {i + 1}: 用户行格式错误")
        if not AI行.startswith('AI:'):
            问题列表.append(f"对话 {i + 1}: AI行格式错误")
        if not 标签行.startswith('标签:'):
            问题列表.append(f"对话 {i + 1}: 标签行格式错误")

        # 提取内容
        用户内容 = 用户行[3:].strip()
        AI内容 = AI行[3:].strip()
        标签 = 标签行[3:].strip()

        # 检查内容是否为空
        if not 用户内容:
            问题列表.append(f"对话 {i + 1}: 用户内容为空")
        if not AI内容:
            问题列表.append(f"对话 {i + 1}: AI内容为空")
        if not 标签:
            问题列表.append(f"对话 {i + 1}: 标签为空")

        # 检查长度
        if len(用户内容) < 2:
            问题列表.append(f"对话 {i + 1}: 用户内容太短")
        if len(AI内容) < 2:
            问题列表.append(f"对话 {i + 1}: AI内容太短")

        # 检查特殊字符（可选）
        if '###' in 用户内容 or '###' in AI内容:
            问题列表.append(f"对话 {i + 1}: 包含特殊标记###")

    # 输出结果
    if 问题列表:
        print(f"⚠️ 发现 {len(问题列表)} 个问题:")
        for 问题 in 问题列表[:10]:  # 只显示前10个
            print(f"  {问题}")
        if len(问题列表) > 10:
            print(f"  ... 还有 {len(问题列表) - 10} 个问题")
    else:
        print("✅ 数据格式检查通过！")

    # 统计标签分布
    print(f"\n=== 标签分布 ===")
    标签统计 = {}
    for 对话 in 对话块:
        行列表 = 对话.split('\n')
        if len(行列表) >= 3:
            标签 = 行列表[2][3:].strip()
            标签统计[标签] = 标签统计.get(标签, 0) + 1

    for 标签, 数量 in 标签统计.items():
        print(f"  {标签}: {数量} 条 ({数量 / len(对话块) * 100:.1f}%)")

    return len(问题列表) == 0


def 合并多个文件(文件列表, 输出文件):
    """合并多个数据文件"""
    print(f"=== 合并文件 ===")

    所有内容 = []
    for 文件 in 文件列表:
        print(f"读取: {文件}")
        with open(文件, 'r', encoding='utf-8') as f:
            内容 = f.read().strip()
            if 内容:
                所有内容.append(内容)

    合并内容 = '\n\n'.join(所有内容)

    with open(输出文件, 'w', encoding='utf-8') as f:
        f.write(合并内容)

    print(f"合并完成！保存到: {输出文件}")
    if 合并内容:
        对话数 = len(合并内容.split('\n\n'))
    else:
        对话数 = 0
    print(f"总对话数: {对话数}")

    # 检查合并后的数据
    检查数据质量(输出文件)


if __name__ == "__main__":
    # 检查我们之前的数据
    检查数据质量('../数据/原始数据/安全对话.txt')

    # 如果有生成的数据，也检查一下
    try:
        检查数据质量('../数据/原始数据/生成的数据.txt')
    except:
        print("\n未找到生成的数据文件，跳过检查")

    # 合并所有数据
    文件列表 = [
        '../数据/原始数据/安全对话.txt',
        '../数据/原始数据/危险对话.txt',
        '../数据/原始数据/日常对话.txt',
        '../数据/原始数据/轻度危险.txt',
        '../数据/原始数据/明显危险.txt',
        '../数据/原始数据/灰色地带.txt',
        '../数据/原始数据/生成的数据.txt'  # 如果有的话
    ]

    # 只合并存在的文件
    存在的文件 = [f for f in 文件列表 if os.path.exists(f)]

    if 存在的文件:
        合并多个文件(存在的文件, '../数据/原始数据/合并数据.txt')
    else:
        print("未找到数据文件，请先创建数据文件")
