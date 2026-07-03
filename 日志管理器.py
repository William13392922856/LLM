"""
日志管理器模块
提供统一的日志记录和管理功能
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class 日志记录器:
    """统一的日志记录器类"""

    # 日志级别映射
    级别映射 = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    def __init__(self,
                 名称: str = '应用日志',
                 日志级别: str = 'INFO',
                 日志目录: Optional[str] = None,
                 日志文件名: Optional[str] = None,
                 控制台输出: bool = True,
                 文件输出: bool = True,
                 日志格式: Optional[str] = None,
                 日期格式: Optional[str] = None,
                 最大文件大小: int = 10 * 1024 * 1024,  # 10MB
                 备份文件数: int = 5):
        """
        初始化日志记录器

        参数:
            名称: 日志记录器名称
            日志级别: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            日志目录: 日志文件存储目录
            日志文件名: 日志文件名
            控制台输出: 是否输出到控制台
            文件输出: 是否输出到文件
            日志格式: 自定义日志格式
            日期格式: 自定义日期格式
            最大文件大小: 单个日志文件最大大小（字节）
            备份文件数: 保留的备份文件数量
        """
        self.名称 = 名称
        self.日志级别 = self._解析日志级别(日志级别)
        self.日志目录 = Path(日志目录) if 日志目录 else Path.cwd() / 'logs'
        self.日志文件名 = 日志文件名 or f"{名称}_{datetime.now().strftime('%Y%m%d')}.log"
        self.控制台输出 = 控制台输出
        self.文件输出 = 文件输出
        self.最大文件大小 = 最大文件大小
        self.备份文件数 = 备份文件数

        # 默认日志格式
        self.日志格式 = 日志格式 or '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        self.日期格式 = 日期格式 or '%Y-%m-%d %H:%M:%S'

        # 创建日志记录器
        self.记录器 = logging.getLogger(名称)
        self.记录器.setLevel(self.日志级别)

        # 避免重复添加处理器
        if not self.记录器.handlers:
            self._初始化处理器()

    def _解析日志级别(self, 级别: str) -> int:
        """解析日志级别"""
        级别 = 级别.upper()
        return self.级别映射.get(级别, logging.INFO)

    def _初始化处理器(self):
        """初始化日志处理器"""
        # 创建格式化器
        格式化器 = logging.Formatter(self.日志格式, datefmt=self.日期格式)

        # 添加控制台处理器
        if self.控制台输出:
            控制台处理器 = logging.StreamHandler(sys.stdout)
            控制台处理器.setLevel(self.日志级别)
            控制台处理器.setFormatter(格式化器)
            self.记录器.addHandler(控制台处理器)

        # 添加文件处理器
        if self.文件输出:
            self._创建日志目录()
            日志文件路径 = self.日志目录 / self.日志文件名

            # 使用按大小轮转的文件处理器
            文件处理器 = RotatingFileHandler(
                日志文件路径,
                maxBytes=self.最大文件大小,
                backupCount=self.备份文件数,
                encoding='utf-8'
            )
            文件处理器.setLevel(self.日志级别)
            文件处理器.setFormatter(格式化器)
            self.记录器.addHandler(文件处理器)

    def _创建日志目录(self):
        """创建日志目录"""
        if not self.日志目录.exists():
            try:
                self.日志目录.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"警告: 无法创建日志目录 {self.日志目录}: {str(e)}")

    def 信息(self, 消息: str, *args, **kwargs):
        """
        记录INFO级别的日志

        参数:
            消息: 日志消息
        """
        self.记录器.info(消息, *args, **kwargs)

    def 警告(self, 消息: str, *args, **kwargs):
        """
        记录WARNING级别的日志

        参数:
            消息: 日志消息
        """
        self.记录器.warning(消息, *args, **kwargs)

    def 错误(self, 消息: str, *args, **kwargs):
        """
        记录ERROR级别的日志

        参数:
            消息: 日志消息
        """
        self.记录器.error(消息, *args, **kwargs)

    def 调试(self, 消息: str, *args, **kwargs):
        """
        记录DEBUG级别的日志

        参数:
            消息: 日志消息
        """
        self.记录器.debug(消息, *args, **kwargs)

    def 严重错误(self, 消息: str, *args, **kwargs):
        """
        记录CRITICAL级别的日志

        参数:
            消息: 日志消息
        """
        self.记录器.critical(消息, *args, **kwargs)

    def 异常(self, 消息: str, *args, **kwargs):
        """
        记录异常信息（包含堆栈跟踪）

        参数:
            消息: 日志消息
        """
        self.记录器.exception(消息, *args, **kwargs)

    def 设置级别(self, 级别: str):
        """
        设置日志级别

        参数:
            级别: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        新级别 = self._解析日志级别(级别)
        self.日志级别 = 新级别
        self.记录器.setLevel(新级别)
        for 处理器 in self.记录器.handlers:
            处理器.setLevel(新级别)

    def 清理旧日志(self, 保留天数: int = 30):
        """
        清理过期的日志文件

        参数:
            保留天数: 保留日志的天数
        """
        if not self.文件输出 or not self.日志目录.exists():
            return

        截止日期 = datetime.now() - timedelta(days=保留天数)
        清理文件数 = 0

        try:
            for 日志文件 in self.日志目录.glob('*.log*'):
                if 日志文件.is_file():
                    # 获取文件修改时间
                    修改时间 = datetime.fromtimestamp(日志文件.stat().st_mtime)
                    if 修改时间 < 截止日期:
                        try:
                            日志文件.unlink()
                            清理文件数 += 1
                        except Exception as e:
                            self.警告(f"无法删除日志文件 {日志文件}: {str(e)}")

            if 清理文件数 > 0:
                self.信息(f"已清理 {清理文件数} 个过期日志文件")

        except Exception as e:
            self.错误(f"清理日志文件时出错: {str(e)}")

    def 获取日志统计(self) -> Dict[str, Any]:
        """
        获取日志统计信息

        返回:
            包含统计信息的字典
        """
        统计信息 = {
            '日志记录器名称': self.名称,
            '日志级别': logging.getLevelName(self.日志级别),
            '日志目录': str(self.日志目录),
            '日志文件名': self.日志文件名,
            '控制台输出': self.控制台输出,
            '文件输出': self.文件输出,
            '处理器数量': len(self.记录器.handlers)
        }

        # 统计日志文件信息
        if self.文件输出 and self.日志目录.exists():
            日志文件列表 = list(self.日志目录.glob('*.log*'))
            总大小 = sum(f.stat().st_size for f in 日志文件列表 if f.is_file())
            统计信息['日志文件数量'] = len(日志文件列表)
            统计信息['日志文件总大小'] = self._格式化大小(总大小)

        return 统计信息

    def _格式化大小(self, 大小: int) -> str:
        """格式化文件大小"""
        for 单位 in ['B', 'KB', 'MB', 'GB']:
            if 大小 < 1024.0:
                return f"{大小:.2f} {单位}"
            大小 /= 1024.0
        return f"{大小:.2f} TB"


class 日志管理器:
    """日志管理器，管理多个日志记录器"""

    def __init__(self, 默认日志目录: Optional[str] = None):
        """
        初始化日志管理器

        参数:
            默认日志目录: 默认的日志存储目录
        """
        self.默认日志目录 = Path(默认日志目录) if 默认日志目录 else Path.cwd() / 'logs'
        self.记录器字典: Dict[str, 日志记录器] = {}

    def 创建记录器(self,
                   名称: str,
                   日志级别: str = 'INFO',
                   日志文件名: Optional[str] = None,
                   **kwargs) -> 日志记录器:
        """
        创建或获取日志记录器

        参数:
            名称: 日志记录器名称
            日志级别: 日志级别
            日志文件名: 日志文件名
            **kwargs: 其他参数传递给日志记录器

        返回:
            日志记录器实例
        """
        if 名称 in self.记录器字典:
            return self.记录器字典[名称]

        记录器 = 日志记录器(
            名称=名称,
            日志级别=日志级别,
            日志目录=str(self.默认日志目录),
            日志文件名=日志文件名 or f"{名称}.log",
            **kwargs
        )

        self.记录器字典[名称] = 记录器
        return 记录器

    def 获取记录器(self, 名称: str) -> Optional[日志记录器]:
        """
        获取已存在的日志记录器

        参数:
            名称: 日志记录器名称

        返回:
            日志记录器实例，不存在则返回None
        """
        return self.记录器字典.get(名称)

    def 清理所有旧日志(self, 保留天数: int = 30):
        """
        清理所有记录器的旧日志

        参数:
            保留天数: 保留日志的天数
        """
        for 记录器 in self.记录器字典.values():
            记录器.清理旧日志(保留天数)

    def 获取所有统计(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有记录器的统计信息

        返回:
            包含所有记录器统计信息的字典
        """
        return {名称: 记录器.获取日志统计() for 名称, 记录器 in self.记录器字典.items()}


# 全局日志管理器实例
_全局管理器 = 日志管理器()


def 获取记录器(名称: str = '应用', 日志级别: str = 'INFO', **kwargs) -> 日志记录器:
    """
    获取日志记录器的便捷函数

    参数:
        名称: 日志记录器名称
        日志级别: 日志级别
        **kwargs: 其他参数

    返回:
        日志记录器实例
    """
    return _全局管理器.创建记录器(名称, 日志级别, **kwargs)


def 信息(消息: str, 名称: str = '应用'):
    """记录INFO级别日志的便捷函数"""
    记录器 = 获取记录器(名称)
    记录器.信息(消息)


def 警告(消息: str, 名称: str = '应用'):
    """记录WARNING级别日志的便捷函数"""
    记录器 = 获取记录器(名称)
    记录器.警告(消息)


def 错误(消息: str, 名称: str = '应用'):
    """记录ERROR级别日志的便捷函数"""
    记录器 = 获取记录器(名称)
    记录器.错误(消息)


def 调试(消息: str, 名称: str = '应用'):
    """记录DEBUG级别日志的便捷函数"""
    记录器 = 获取记录器(名称)
    记录器.调试(消息)


if __name__ == "__main__":
    # 测试代码
    print("日志管理器测试")
    print("=" * 50)
    
    # 创建日志记录器
    日志 = 日志记录器(
        名称='测试日志',
        日志级别='DEBUG',
        日志目录='./logs',
        日志文件名='test.log'
    )
    
    # 测试各级别日志
    日志.调试("这是一条调试信息")
    日志.信息("这是一条普通信息")
    日志.警告("这是一条警告信息")
    日志.错误("这是一条错误信息")
    
    # 测试异常记录
    try:
        1 / 0
    except Exception as e:
        日志.异常("发生异常")
    
    # 显示统计信息
    统计 = 日志.获取日志统计()
    print("\n日志统计信息:")
    for 键, 值 in 统计.items():
        print(f"  {键}: {值}")
    
    print("\n日志管理器测试完成")