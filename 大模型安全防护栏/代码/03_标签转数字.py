import pandas as pd

# 读取刚才保存的数据
df = pd.read_csv('../数据/处理后的数据/对话数据.csv')

# 显示前几行
print("前几行原始数据：")
print(df.head())

# 定义一个标签到数字的映射
label_mapping = {
    '安全': 0,
    '危险-暴力': 1,
    '危险-粗俗': 2,
    '危险-违法': 3,
    '危险-自残': 4
}

# 添加数字标签列
df['标签数字'] = df['标签'].map(label_mapping)

print("\n添加数字标签后：")
print(df[['用户', 'AI', '标签', '标签数字']].head())

# 反向映射（数字转文字，用于查看）
reverse_mapping = {v: k for k, v in label_mapping.items()}
print(f"\n标签映射关系：")
for num, text in reverse_mapping.items():
    print(f"  {num}: {text}")

# 保存新版本
df.to_csv('../数据/处理后的数据/对话数据_带数字标签.csv', index=False, encoding='utf-8-sig')
print("\n已保存新文件！")