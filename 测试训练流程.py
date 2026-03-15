"""测试完整的训练流程"""
import subprocess
import sys

print("开始测试训练流程...")

# 运行训练器并自动输入 'y' 开始训练
process = subprocess.Popen(
    [sys.executable, '源代码\\模型训练器.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd='.'
)

# 发送 'y' 并回车
process.stdin.write('y\n')
process.stdin.flush()

# 读取输出
stdout, stderr = process.communicate()

print("\n训练器输出:")
print(stdout)

if stderr:
    print("\n错误信息:")
    print(stderr)

print(f"\n训练器退出码: {process.returncode}")
