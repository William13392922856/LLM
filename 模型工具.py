"""
模型工具模块
提供统一的模型加载、验证和管理功能
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class 模型工具类:
    """统一的模型加载工具类"""

    # 支持的模型类型
    支持的模型类型 = {
        'pytorch': ['.pt', '.pth', '.bin'],
        'safetensors': ['.safetensors'],
        'tensorflow': ['.h5', '.pb'],
        'onnx': ['.onnx'],
        'gguf': ['.gguf']
    }

    def __init__(self, 基础路径: Optional[str] = None):
        """
        初始化模型工具类

        参数:
            基础路径: 模型存储的基础路径，默认为当前目录
        """
        self.基础路径 = Path(基础路径) if 基础路径 else Path.cwd()
        self.模型信息缓存: Dict[str, Dict[str, Any]] = {}

    def 查找可用模型(self, 搜索路径: Optional[str] = None, 
                    递归搜索: bool = True) -> List[Dict[str, Any]]:
        """
        查找指定路径下的所有可用模型

        参数:
            搜索路径: 要搜索的路径，默认使用基础路径
            递归搜索: 是否递归搜索子目录

        返回:
            包含模型信息的字典列表
        """
        搜索路径 = Path(搜索路径) if 搜索路径 else self.基础路径
        模型列表 = []

        try:
            if not 搜索路径.exists():
                print(f"警告: 搜索路径不存在: {搜索路径}")
                return 模型列表

            搜索模式 = '**/*' if 递归搜索 else '*'

            for 文件路径 in 搜索路径.glob(搜索模式):
                if 文件路径.is_file():
                    模型信息 = self.识别模型类型(文件路径)
                    if 模型信息:
                        模型列表.append(模型信息)

        except Exception as e:
            print(f"错误: 搜索模型时发生异常: {str(e)}")

        return 模型列表

    def 识别模型类型(self, 模型路径: Path) -> Optional[Dict[str, Any]]:
        """
        识别模型文件类型

        参数:
            模型路径: 模型文件路径

        返回:
            包含模型信息的字典，如果无法识别则返回None
        """
        try:
            模型路径 = Path(模型路径)
            if not 模型路径.exists():
                return None

            文件后缀 = 模型路径.suffix.lower()
            文件名 = 模型路径.stem
            文件大小 = 模型路径.stat().st_size

            # 识别模型类型
            模型类型 = None
            for 类型, 后缀列表 in self.支持的模型类型.items():
                if 文件后缀 in 后缀列表:
                    模型类型 = 类型
                    break

            if 模型类型 is None:
                return None

            模型信息 = {
                '路径': str(模型路径),
                '文件名': 文件名,
                '类型': 模型类型,
                '后缀': 文件后缀,
                '大小': 文件大小,
                '大小格式化': self._格式化文件大小(文件大小)
            }

            # 尝试加载模型配置文件
            配置信息 = self._加载模型配置(模型路径)
            if 配置信息:
                模型信息['配置'] = 配置信息

            return 模型信息

        except Exception as e:
            print(f"警告: 识别模型类型时出错 {模型路径}: {str(e)}")
            return None

    def 加载模型信息(self, 模型路径: str, 
                    强制重新加载: bool = False) -> Optional[Dict[str, Any]]:
        """
        加载模型的详细信息

        参数:
            模型路径: 模型文件路径
            强制重新加载: 是否强制重新加载（忽略缓存）

        返回:
            包含详细模型信息的字典
        """
        模型路径 = str(模型路径)

        # 检查缓存
        if not 强制重新加载 and 模型路径 in self.模型信息缓存:
            return self.模型信息缓存[模型路径]

        try:
            路径对象 = Path(模型路径)
            if not 路径对象.exists():
                raise FileNotFoundError(f"模型文件不存在: {模型路径}")

            # 获取基础模型信息
            模型信息 = self.识别模型类型(路径对象)
            if 模型信息 is None:
                raise ValueError(f"无法识别模型类型: {模型路径}")

            # 添加额外信息
            模型信息['绝对路径'] = str(路径对象.absolute())
            模型信息['相对路径'] = str(路径对象.relative_to(self.基础路径)) if 路径对象.is_relative_to(self.基础路径) else 模型路径

            # 验证模型完整性
            模型信息['完整性验证'] = self.验证模型完整性(模型路径)

            # 缓存结果
            self.模型信息缓存[模型路径] = 模型信息

            return 模型信息

        except Exception as e:
            print(f"错误: 加载模型信息失败: {str(e)}")
            return None

    def 验证模型完整性(self, 模型路径: str) -> Dict[str, Any]:
        """
        验证模型文件的完整性

        参数:
            模型路径: 模型文件路径

        返回:
            包含验证结果的字典
        """
        验证结果 = {
            '有效': False,
            '文件存在': False,
            '文件可读': False,
            '文件大小正常': False,
            '配置文件存在': False,
            '错误信息': []
        }

        try:
            路径对象 = Path(模型路径)

            # 检查文件是否存在
            验证结果['文件存在'] = 路径对象.exists()
            if not 验证结果['文件存在']:
                验证结果['错误信息'].append("模型文件不存在")
                return 验证结果

            # 检查文件是否可读
            验证结果['文件可读'] = os.access(模型路径, os.R_OK)
            if not 验证结果['文件可读']:
                验证结果['错误信息'].append("模型文件不可读")

            # 检查文件大小（至少1KB）
            文件大小 = 路径对象.stat().st_size
            验证结果['文件大小正常'] = 文件大小 > 1024
            if not 验证结果['文件大小正常']:
                验证结果['错误信息'].append(f"模型文件过小 ({文件大小} bytes)")

            # 检查配置文件
            配置文件 = self._查找配置文件(路径对象)
            验证结果['配置文件存在'] = 配置文件 is not None
            验证结果['配置文件路径'] = str(配置文件) if 配置文件 else None

            # 总体判断
            验证结果['有效'] = (
                验证结果['文件存在'] and 
                验证结果['文件可读'] and 
                验证结果['文件大小正常']
            )

        except Exception as e:
            验证结果['错误信息'].append(f"验证过程出错: {str(e)}")

        return 验证结果

    def 构建模型路径(self, 模型名称: str, 
                    模型类型: Optional[str] = None,
                    子目录: Optional[str] = None) -> Path:
        """
        构建模型文件的完整路径

        参数:
            模型名称: 模型名称
            模型类型: 模型类型（用于确定文件后缀）
            子目录: 模型所在的子目录

        返回:
            构建的路径对象
        """
        # 确定基础路径
        目标路径 = self.基础路径
        if 子目录:
            目标路径 = 目标路径 / 子目录

        # 确定文件后缀
        if 模型类型 and 模型类型 in self.支持的模型类型:
            # 使用该类型的第一个后缀作为默认
            后缀 = self.支持的模型类型[模型类型][0]
        else:
            后缀 = ''

        # 构建完整路径
        if not 模型名称.endswith(后缀):
            文件名 = f"{模型名称}{后缀}"
        else:
            文件名 = 模型名称

        return 目标路径 / 文件名

    def _格式化文件大小(self, 大小: int) -> str:
        """格式化文件大小为易读格式"""
        for 单位 in ['B', 'KB', 'MB', 'GB', 'TB']:
            if 大小 < 1024.0:
                return f"{大小:.2f} {单位}"
            大小 /= 1024.0
        return f"{大小:.2f} PB"

    def _加载模型配置(self, 模型路径: Path) -> Optional[Dict[str, Any]]:
        """加载模型的配置文件"""
        配置文件名列表 = [
            'config.json',
            'model_config.json',
            'configuration.json',
            'params.json'
        ]

        # 检查同目录下的配置文件
        模型目录 = 模型路径.parent if 模型路径.is_file() else 模型路径

        for 配置文件名 in 配置文件名列表:
            配置路径 = 模型目录 / 配置文件名
            if 配置路径.exists():
                try:
                    with open(配置路径, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"警告: 无法加载配置文件 {配置路径}: {str(e)}")

        return None

    def _查找配置文件(self, 模型路径: Path) -> Optional[Path]:
        """查找模型的配置文件"""
        配置文件名列表 = [
            'config.json',
            'model_config.json',
            'configuration.json',
            'params.json'
        ]

        模型目录 = 模型路径.parent if 模型路径.is_file() else 模型路径

        for 配置文件名 in 配置文件名列表:
            配置路径 = 模型目录 / 配置文件名
            if 配置路径.exists():
                return 配置路径

        return None


# 便捷函数
def 查找可用模型(搜索路径: Optional[str] = None, 递归搜索: bool = True) -> List[Dict[str, Any]]:
    """
    查找可用模型的便捷函数

    参数:
        搜索路径: 要搜索的路径
        递归搜索: 是否递归搜索

    返回:
        模型信息列表
    """
    工具实例 = 模型工具类(搜索路径)
    return 工具实例.查找可用模型(递归搜索=递归搜索)


def 识别模型类型(模型路径: str) -> Optional[Dict[str, Any]]:
    """
    识别模型类型的便捷函数

    参数:
        模型路径: 模型文件路径

    返回:
        模型信息字典
    """
    工具实例 = 模型工具类()
    return 工具实例.识别模型类型(Path(模型路径))


def 验证模型完整性(模型路径: str) -> Dict[str, Any]:
    """
    验证模型完整性的便捷函数

    参数:
        模型路径: 模型文件路径

    返回:
        验证结果字典
    """
    工具实例 = 模型工具类()
    return 工具实例.验证模型完整性(模型路径)


if __name__ == "__main__":
    # 测试代码
    print("模型工具模块测试")
    print("=" * 50)
    
    # 创建测试实例
    工具 = 模型工具类()
    
    # 测试路径构建
    测试路径 = 工具.构建模型路径("test_model", "pytorch", "models")
    print(f"构建的模型路径: {测试路径}")
    
    print("\n支持的模型类型:")
    for 类型, 后缀列表 in 模型工具类.支持的模型类型.items():
        print(f"  {类型}: {', '.join(后缀列表)}")