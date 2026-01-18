"""
安全防护栏模型训练系统2

在PyCharm中直接运行
"""

import sys
import os
import json
from pathlib import Path
import yaml

# 设置路径
当前目录 = Path(__file__).parent
源代码目录 = 当前目录 / "源代码"
sys.path.append(str(源代码目录))

def 主程序():
    """主程序入口"""
    print("="*60)
    print("安全防护栏模型训练系统")
    print("="*60)

    # 1. 检查目录结构
    检查目录()

    # 2. 检查配置文件
    配置文件 = 检查配置文件()

    while True:
        显示主菜单()
        选择 = input("\n请选择操作 (1-7): ").strip()

        if 选择 == "1":
            运行样本管理()

        elif 选择 == "2":
            运行数据收集(配置文件)

        elif 选择 == "3":
            运行数据处理()

        elif 选择 == "4":
            运行模型训练(配置文件)

        elif 选择 == "5":
            运行模型测试()

        elif 选择 == "6":
            查看数据报告()

        elif 选择 == "7":
            系统工具(配置文件)

        elif 选择 == "0":
            print("👋 感谢使用，再见！")
            break

        else:
            print("❌ 无效选项")

def 检查目录():
    """检查并创建必要的目录"""
    print("\n🔍 检查项目结构...")

    必要目录 = [
        "数据/原始数据",
        "数据/处理数据",
        "源代码",
        "配置文件",
        "模型文件",
        "日志文件"
    ]

    for 相对路径 in 必要目录:
        目录路径 = 当前目录 / 相对路径
        if not 目录路径.exists():
            目录路径.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ 创建: {相对路径}")

    print("✅ 目录结构检查完成")

def 检查配置文件():
    """检查并创建配置文件"""
    配置路径 = 当前目录 / "配置文件" / "配置.yaml"

    if not 配置路径.exists():
        print("\n⚠️  正在创建配置文件...")

        默认配置 = {
            "api": {
                "base_url": "https://api.deepseek.com",
                "api_key": "sk-28bc3150642c44daaa4d7e5489af1455",
                "model": "deepseek-chat"
            },
            "model": {
                "base_model": "Qwen/Qwen2.5-1.5B-Instruct"
            },
            "training": {
                "learning_rate": 2e-4,
                "num_epochs": 3
            }
        }

        with open(配置路径, 'w', encoding='utf-8') as f:
            yaml.dump(默认配置, f, default_flow_style=False, allow_unicode=True)

        print(f"✅ 配置文件已创建: {配置路径}")

    # 读取配置
    with open(配置路径, 'r', encoding='utf-8') as f:
        配置 = yaml.safe_load(f)

    return 配置

def 显示主菜单():
    """显示主菜单"""
    print("\n" + "="*50)
    print("📋 主菜单")
    print("="*50)
    print("1. 📝 样本管理 - 添加/编辑攻击性提示词")
    print("2. 🤖 收集推理 - 从DeepSeek获取安全分析")
    print("3. 🔧 处理数据 - 智能处理样本生成训练集")
    print("4. 🎯 训练模型 - 训练安全防护栏模型")
    print("5. 🧪 测试模型 - 测试模型安全判断")
    print("6. 📊 数据报告 - 查看数据统计")
    print("7. ⚙️  系统工具 - 配置和工具")
    print("0. ❌ 退出系统")
    print("="*50)

def 运行样本管理():
    """运行样本管理工具"""
    print("\n" + "="*50)
    print("📝 样本管理系统")
    print("="*50)

    try:
        from 样本管理工具 import 样本管理工具
        工具 = 样本管理工具()
        工具.运行()
    except ImportError as e:
        print(f"❌ 错误: 无法导入样本管理工具")
        print(f"请确保 源代码/样本管理工具.py 存在")

def 运行数据收集(配置):
    """运行数据收集"""
    print("\n" + "="*50)
    print("🤖 推理数据收集")
    print("="*50)

    # 检查API密钥
    api_key = 配置.get('api', {}).get('api_key', '')
    if not api_key or api_key == "在这里填入你的API密钥":
        print("❌ 请先配置API密钥")
        print("请到 配置文件/配置.yaml 中填写正确的API密钥")
        return

    try:
        from 数据收集器 import 数据收集器

        样本数 = input("要收集多少个样本? (默认100): ").strip()
        样本数 = int(样本数) if 样本数 else 100

        print(f"\n开始收集 {样本数} 个样本的推理数据...")
        print("这可能需要一些时间，请耐心等待...")

        收集器 = 数据收集器(API密钥=api_key)
        数据集 = 收集器.从文本收集(样本数=样本数)

        print(f"\n✅ 推理数据收集完成！")
        print(f"共收集 {len(数据集)} 个样本")

    except ImportError:
        print("❌ 错误: 无法导入数据收集器")
    except Exception as e:
        print(f"❌ 收集数据时出错: {e}")

def 运行数据处理():
    """运行数据处理"""
    print("\n" + "="*50)
    print("🔧 智能数据处理")
    print("="*50)

    try:
        from 智能数据处理器 import 智能数据处理器

        print("正在智能处理数据...")
        print("支持格式: 带标签[危险]/[安全]、带序号、纯文本")

        处理器 = 智能数据处理器()
        结果 = 处理器.运行()

        if 结果:
            print("\n✅ 数据处理完成！")
            print("生成的训练数据在: 数据/处理数据/")
        else:
            print("\n❌ 数据处理失败")

    except ImportError:
        print("❌ 错误: 无法导入智能数据处理器")

def 运行模型训练(配置):
    """运行模型训练"""
    print("\n" + "="*50)
    print("🎯 模型训练")
    print("="*50)

    # 检查训练数据
    训练数据路径 = 当前目录 / "数据" / "处理数据" / "训练集.json"
    if not 训练数据路径.exists():
        print("❌ 找不到训练数据")
        print("请先运行 '处理数据' 生成训练集")
        return

    try:
        from 模型训练器 import 模型训练器

        print("开始训练安全防护栏模型...")
        print("这可能需要较长时间，请耐心等待...")

        训练器 = 模型训练器()
        训练器.开始训练()

        print("\n✅ 模型训练完成！")
        print("模型已保存到: 模型文件/")

    except ImportError:
        print("❌ 错误: 无法导入模型训练器")
        print("请创建 源代码/模型训练器.py 文件")
    except Exception as e:
        print(f"❌ 训练出错: {e}")

def 运行模型测试():
    """运行模型测试"""
    print("\n" + "="*50)
    print("🧪 模型测试")
    print("="*50)

    模型目录 = 当前目录 / "模型文件"
    if not 模型目录.exists() or len(list(模型目录.glob("*"))) == 0:
        print("❌ 没有找到训练好的模型")
        print("请先运行 '训练模型'")
        return

    print("模型测试功能开发中...")
    print("\n已训练模型列表:")

    模型列表 = list(模型目录.glob("*"))
    for i, 模型路径 in enumerate(模型列表, 1):
        print(f"  {i}. {模型路径.name}")

def 查看数据报告():
    """查看数据报告"""
    print("\n" + "="*50)
    print("📊 数据报告")
    print("="*50)

    报告文件 = 当前目录 / "数据" / "处理数据" / "数据报告.txt"

    if 报告文件.exists():
        with open(报告文件, 'r', encoding='utf-8') as f:
            内容 = f.read()
        print(内容)
    else:
        print("❌ 没有找到数据报告")
        print("请先运行 '处理数据'")

    # 显示数据文件列表
    数据目录 = 当前目录 / "数据" / "处理数据"
    if 数据目录.exists():
        文件列表 = [f for f in 数据目录.glob("*.json") if f.is_file()]
        if 文件列表:
            print("\n📁 数据文件:")
            for 文件 in 文件列表:
                大小 = 文件.stat().st_size
                print(f"  - {文件.name} ({大小/1024:.1f} KB)")

def 系统工具(配置):
    """系统工具"""
    print("\n" + "="*50)
    print("⚙️ 系统工具")
    print("="*50)

    while True:
        print("\n工具选项:")
        print("1. 配置API密钥")
        print("2. 清理临时文件")
        print("3. 查看项目结构")
        print("4. 返回主菜单")

        选择 = input("\n请选择 (1-4): ").strip()

        if 选择 == "1":
            配置API密钥(配置)
        elif 选择 == "2":
            清理临时文件()
        elif 选择 == "3":
            查看项目结构()
        elif 选择 == "4":
            break
        else:
            print("❌ 无效选项")

def 配置API密钥(配置):
    """配置API密钥"""
    配置路径 = 当前目录 / "配置文件" / "配置.yaml"

    print(f"\n当前API密钥: {配置.get('api', {}).get('api_key', '未设置')}")
    新密钥 = input("请输入新的API密钥: ").strip()

    if 新密钥:
        配置['api']['api_key'] = 新密钥

        with open(配置路径, 'w', encoding='utf-8') as f:
            yaml.dump(配置, f, default_flow_style=False, allow_unicode=True)

        print("✅ API密钥已更新")
    else:
        print("❌ 密钥不能为空")

def 清理临时文件():
    """清理临时文件"""
    import shutil

    临时目录 = 当前目录 / "临时文件"
    日志目录 = 当前目录 / "日志文件"

    删除数量 = 0
    总大小 = 0

    for 目录 in [临时目录, 日志目录]:
        if 目录.exists():
            for 文件 in 目录.glob("*"):
                if 文件.is_file():
                    大小 = 文件.stat().st_size
                    文件.unlink()
                    删除数量 += 1
                    总大小 += 大小

    总大小_mb = 总大小 / 1024 / 1024
    print(f"\n✅ 已清理 {删除数量} 个文件")
    print(f"释放空间: {总大小_mb:.2f} MB")

def 查看项目结构():
    """查看项目结构"""
    print("\n📁 项目结构:")
    print(f"项目根目录: {当前目录}")

    for 项目 in 当前目录.iterdir():
        if 项目.is_dir() and 项目.name not in [".venv", ".idea", "__pycache__"]:
            # 计算目录大小
            文件数量 = sum(1 for _ in 项目.rglob("*") if _.is_file())
            总大小 = sum(f.stat().st_size for f in 项目.rglob("*") if f.is_file())
            总大小_mb = 总大小 / 1024 / 1024

            print(f"\n  📁 {项目.name}/")
            print(f"    文件数: {文件数量}")
            print(f"    大小: {总大小_mb:.1f} MB")

if __name__ == "__main__":
    try:
        主程序()
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
