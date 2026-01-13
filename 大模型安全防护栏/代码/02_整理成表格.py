import pandas as pd  # 数据处理库


# 定义一个函数来解析对话
def parse_dialogue(dialogue_text):
    lines = dialogue_text.split('\n')
    data = {'用户': '', 'AI': '', '标签': ''}

    for line in lines:
        if line.startswith('用户:'):
            data['用户'] = line[3:].strip()
        elif line.startswith('AI:'):
            data['AI'] = line[3:].strip()
        elif line.startswith('标签:'):
            data['标签'] = line[3:].strip()

    return data


# 读取所有对话
all_dialogues = []

# 读取合并数据文件
with open('../数据/原始数据/合并数据.txt', 'r', encoding='utf-8') as f:
    dialogues = f.read().strip().split('\n\n')
    for dialogue in dialogues:
        all_dialogues.append(parse_dialogue(dialogue))

# 转换成表格
df = pd.DataFrame(all_dialogues)
print("整理后的数据表格：")
print(df)
print("\n表格信息：")
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")

# 统计标签分布
print("\n各个标签的数量：")
print(df['标签'].value_counts())

# 保存为CSV文件
df.to_csv('../数据/处理后的数据/对话数据.csv', index=False, encoding='utf-8-sig')
print("\n已保存为 CSV 文件！")