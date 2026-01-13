import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 指定默认字体为黑体或微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 读取数据
df = pd.read_csv('../数据/处理后的数据/对话数据_带数字标签.csv')

print("=== 数据基本信息 ===")
print(f"总样本数: {len(df)}")
print(f"列名: {list(df.columns)}")

print("\n=== 标签分布 ===")
标签统计 = df['标签'].value_counts()
print(标签统计)

# 计算文本长度
df['用户长度'] = df['用户'].apply(lambda x: len(str(x)))
df['AI长度'] = df['AI'].apply(lambda x: len(str(x)))

print("\n=== 文本长度统计 ===")
print(f"用户输入平均长度: {df['用户长度'].mean():.1f} 字符")
print(f"AI回复平均长度: {df['AI长度'].mean():.1f} 字符")
print(f"用户输入最长: {df['用户长度'].max()} 字符")
print(f"AI回复最长: {df['AI长度'].max()} 字符")

# 保存详细统计
统计结果 = {
    '总样本数': len(df),
    '用户平均长度': round(df['用户长度'].mean(), 1),
    'AI平均长度': round(df['AI长度'].mean(), 1),
    '标签分布': 标签统计.to_dict()
}

import json

with open('../数据/处理后的数据/数据统计.json', 'w', encoding='utf-8') as f:
    json.dump(统计结果, f, ensure_ascii=False, indent=2)

print("\n已保存统计信息到 JSON 文件！")

# 简单的可视化（需要安装matplotlib）
try:
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 标签分布饼图
    标签统计.plot.pie(ax=axes[0], autopct='%1.1f%%')
    axes[0].set_title('安全标签分布')
    axes[0].set_ylabel('')

    # 文本长度直方图
    axes[1].hist(df['用户长度'], bins=10, alpha=0.7, label='用户输入')
    axes[1].hist(df['AI长度'], bins=10, alpha=0.7, label='AI回复')
    axes[1].set_xlabel('文本长度（字符）')
    axes[1].set_ylabel('数量')
    axes[1].set_title('文本长度分布')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('../数据/处理后的数据/数据统计图.png', dpi=100)
    print("已保存统计图表！")
except:
    print("如需生成图表，请运行: pip install matplotlib")