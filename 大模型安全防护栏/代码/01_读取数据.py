# 第一步：读取文件
print("=== 开始读取数据 ===")

# 打开文件
with open('../数据/原始数据/合并数据.txt.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print("文件内容：")
    print(content)
    print("=" * 50)

# 第二步：把内容分割成对话
dialogues = content.strip().split('\n\n')  # 用空行分割
print(f"总共找到了 {len(dialogues)} 个对话")

# 第三步：解析每个对话
for i, dialogue in enumerate(dialogues):
    print(f"\n对话 {i + 1}:")
    lines = dialogue.split('\n')

    for line in lines:
        if line.startswith('用户:'):
            print(f"  用户说: {line[3:]}")  # 去掉"用户:"三个字
        elif line.startswith('AI:'):
            print(f"  AI回复: {line[3:]}")  # 去掉"AI:"三个字
        elif line.startswith('标签:'):
            print(f"  安全标签: {line[3:]}")  # 去掉"标签:"三个字