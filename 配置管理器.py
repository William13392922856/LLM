"""
配置管理器模块
提供统一的配置加载、验证和管理功能
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from copy import deepcopy


class 配置错误(Exception):
    """配置相关错误的异常类"""
    pass


class 配置管理器:
    """统一的配置管理类"""

    支持的配置格式 = ['.json', '.yaml', '.yml', '.toml']

    def __init__(self, 配置文件路径: Optional[str] = None, 默认配置: Optional[Dict] = None):
        """
        初始化配置管理器

        参数:
            配置文件路径: 配置文件的路径
            默认配置: 默认配置字典
        """
        self.配置数据: Dict[str, Any] = {}
        self.配置文件路径: Optional[Path] = None
        self.默认配置 = 默认配置 or {}
        self.已加载 = False

        if 配置文件路径:
            self.加载配置(配置文件路径)

    def 加载配置(self, 配置文件路径: str, 合并默认配置: bool = True) -> Dict[str, Any]:
        """
        从文件加载配置

        参数:
            配置文件路径: 配置文件路径
            合并默认配置: 是否与默认配置合并

        返回:
            配置字典

        异常:
            配置错误: 当配置文件不存在或格式错误时抛出
        """
        路径对象 = Path(配置文件路径)

        if not 路径对象.exists():
            raise 配置错误(f"配置文件不存在: {配置文件路径}")

        文件后缀 = 路径对象.suffix.lower()

        try:
            with open(路径对象, 'r', encoding='utf-8') as f:
                if 文件后缀 == '.json':
                    加载的配置 = json.load(f)
                elif 文件后缀 in ['.yaml', '.yml']:
                    加载的配置 = yaml.safe_load(f)
                elif 文件后缀 == '.toml':
                    import toml
                    加载的配置 = toml.load(f)
                else:
                    raise 配置错误(f"不支持的配置文件格式: {文件后缀}")

            # 合并默认配置
            if 合并默认配置 and self.默认配置:
                加载的配置 = self._合并配置(self.默认配置, 加载的配置)

            self.配置数据 = 加载的配置
            self.配置文件路径 = 路径对象
            self.已加载 = True

            return self.配置数据

        except json.JSONDecodeError as e:
            raise 配置错误(f"JSON格式错误: {str(e)}")
        except yaml.YAMLError as e:
            raise 配置错误(f"YAML格式错误: {str(e)}")
        except Exception as e:
            raise 配置错误(f"加载配置文件失败: {str(e)}")

    def 验证配置完整性(self, 必需键列表: List[str], 
                      配置: Optional[Dict] = None) -> Dict[str, Any]:
        """
        验证配置的完整性

        参数:
            必需键列表: 必需存在的配置键列表
            配置: 要验证的配置，默认使用当前加载的配置

        返回:
            包含验证结果的字典
        """
        验证结果 = {
            '有效': True,
            '缺失键': [],
            '空值键': [],
            '错误信息': []
        }

        待验证配置 = 配置 if 配置 is not None else self.配置数据

        if not 待验证配置:
            验证结果['有效'] = False
            验证结果['错误信息'].append("配置为空或未加载")
            return 验证结果

        # 检查必需键
        for 键 in 必需键列表:
            键值 = self._获取嵌套值(待验证配置, 键)
            
            if 键值 is None:
                验证结果['缺失键'].append(键)
                验证结果['有效'] = False
            elif isinstance(键值, str) and not 键值.strip():
                验证结果['空值键'].append(键)
                验证结果['错误信息'].append(f"配置项 '{键}' 的值为空")

        return 验证结果

    def 安全获取配置项(self, 键: str, 默认值: Any = None, 
                       配置: Optional[Dict] = None) -> Any:
        """
        安全获取配置项，支持嵌套键（使用点号分隔）

        参数:
            键: 配置键，支持嵌套（如 'database.host'）
            默认值: 当键不存在时的默认返回值
            配置: 要获取的配置，默认使用当前加载的配置

        返回:
            配置项的值或默认值
        """
        待查询配置 = 配置 if 配置 is not None else self.配置数据
        
        if not 待查询配置:
            return 默认值

        值 = self._获取嵌套值(待查询配置, 键)
        return 值 if 值 is not None else 默认值

    def 更新配置(self, 键: str, 值: Any, 立即保存: bool = False) -> bool:
        """
        更新配置项

        参数:
            键: 配置键，支持嵌套（如 'database.host'）
            值: 新的配置值
            立即保存: 是否立即保存到文件

        返回:
            更新是否成功
        """
        try:
            键列表 = 键.split('.')
            当前配置 = self.配置数据

            # 遍历到倒数第二层
            for 当前键 in 键列表[:-1]:
                if 当前键 not in 当前配置:
                    当前配置[当前键] = {}
                当前配置 = 当前配置[当前键]

            # 设置最终值
            当前配置[键列表[-1]] = 值

            if 立即保存:
                return self.保存配置()

            return True

        except Exception as e:
            print(f"错误: 更新配置失败: {str(e)}")
            return False

    def 批量更新配置(self, 更新字典: Dict[str, Any], 立即保存: bool = False) -> bool:
        """
        批量更新配置

        参数:
            更新字典: 要更新的配置键值对字典
            立即保存: 是否立即保存到文件

        返回:
            更新是否成功
        """
        try:
            for 键, 值 in 更新字典.items():
                self.更新配置(键, 值, 立即保存=False)

            if 立即保存:
                return self.保存配置()

            return True

        except Exception as e:
            print(f"错误: 批量更新配置失败: {str(e)}")
            return False

    def 保存配置(self, 文件路径: Optional[str] = None, 
                格式: Optional[str] = None) -> bool:
        """
        保存配置到文件

        参数:
            文件路径: 保存路径，默认使用当前配置文件路径
            格式: 保存格式（json/yaml/toml），默认根据文件扩展名判断

        返回:
            保存是否成功
        """
        保存路径 = Path(文件路径) if 文件路径 else self.配置文件路径

        if 保存路径 is None:
            print("错误: 未指定保存路径")
            return False

        try:
            # 确保目录存在
            保存路径.parent.mkdir(parents=True, exist_ok=True)

            # 确定格式
            if 格式:
                文件后缀 = f'.{格式.lower()}'
            else:
                文件后缀 = 保存路径.suffix.lower()

            # 根据格式保存
            with open(保存路径, 'w', encoding='utf-8') as f:
                if 文件后缀 == '.json':
                    json.dump(self.配置数据, f, ensure_ascii=False, indent=2)
                elif 文件后缀 in ['.yaml', '.yml']:
                    yaml.dump(self.配置数据, f, allow_unicode=True, default_flow_style=False)
                elif 文件后缀 == '.toml':
                    import toml
                    toml.dump(self.配置数据, f)
                else:
                    print(f"错误: 不支持的配置文件格式: {文件后缀}")
                    return False

            print(f"配置已保存到: {保存路径}")
            return True

        except Exception as e:
            print(f"错误: 保存配置失败: {str(e)}")
            return False

    def 重新加载配置(self) -> bool:
        """
        重新从文件加载配置

        返回:
            重新加载是否成功
        """
        if self.配置文件路径 and self.配置文件路径.exists():
            try:
                self.加载配置(str(self.配置文件路径))
                return True
            except Exception as e:
                print(f"错误: 重新加载配置失败: {str(e)}")
                return False
        return False

    def 获取所有配置(self) -> Dict[str, Any]:
        """
        获取所有配置数据

        返回:
            配置字典的深拷贝
        """
        return deepcopy(self.配置数据)

    def 设置默认配置(self, 默认配置: Dict[str, Any], 合并到当前: bool = False):
        """
        设置默认配置

        参数:
            默认配置: 默认配置字典
            合并到当前: 是否将默认配置合并到当前配置
        """
        self.默认配置 = 默认配置
        
        if 合并到当前:
            self.配置数据 = self._合并配置(默认配置, self.配置数据)

    def _获取嵌套值(self, 配置: Dict, 键: str) -> Any:
        """
        从嵌套字典中获取值

        参数:
            配置: 配置字典
            键: 点号分隔的键路径

        返回:
            找到的值或None
        """
        键列表 = 键.split('.')
        当前值 = 配置

        try:
            for 当前键 in 键列表:
                当前值 = 当前值[当前键]
            return 当前值
        except (KeyError, TypeError):
            return None

    def _合并配置(self, 基础配置: Dict, 覆盖配置: Dict) -> Dict:
        """
        深度合并两个配置字典

        参数:
            基础配置: 基础配置
            覆盖配置: 要覆盖的配置

        返回:
            合并后的配置字典
        """
        结果 = deepcopy(基础配置)
        
        for 键, 值 in 覆盖配置.items():
            if 键 in 结果 and isinstance(结果[键], dict) and isinstance(值, dict):
                结果[键] = self._合并配置(结果[键], 值)
            else:
                结果[键] = deepcopy(值)

        return 结果

    def __getitem__(self, 键: str) -> Any:
        """支持字典式访问"""
        return self.安全获取配置项(键)

    def __setitem__(self, 键: str, 值: Any):
        """支持字典式设置"""
        self.更新配置(键, 值)

    def __contains__(self, 键: str) -> bool:
        """支持 'in' 操作符"""
        return self._获取嵌套值(self.配置数据, 键) is not None

    def __repr__(self) -> str:
        return f"配置管理器(已加载={self.已加载}, 配置项数量={len(self.配置数据)})"


# 便捷函数
def 加载配置(配置文件路径: str, 默认配置: Optional[Dict] = None) -> 配置管理器:
    """
    加载配置文件的便捷函数

    参数:
        配置文件路径: 配置文件路径
        默认配置: 默认配置字典

    返回:
        配置管理器实例
    """
    return 配置管理器(配置文件路径, 默认配置)


def 验证配置完整性(配置: Dict, 必需键列表: List[str]) -> Dict[str, Any]:
    """
    验证配置完整性的便捷函数

    参数:
        配置: 配置字典
        必需键列表: 必需的配置键列表

    返回:
        验证结果字典
    """
    管理器 = 配置管理器()
    管理器.配置数据 = 配置
    return 管理器.验证配置完整性(必需键列表)


def 安全获取配置项(配置: Dict, 键: str, 默认值: Any = None) -> Any:
    """
    安全获取配置项的便捷函数

    参数:
        配置: 配置字典
        键: 配置键
        默认值: 默认值

    返回:
        配置项的值
    """
    管理器 = 配置管理器()
    管理器.配置数据 = 配置
    return 管理器.安全获取配置项(键, 默认值)


if __name__ == "__main__":
    # 测试代码
    import tempfile
    
    print("配置管理器测试")
    print("=" * 50)
    
    # 创建测试配置文件
    测试配置 = {
        "应用名称": "测试应用",
        "版本": "1.0.0",
        "数据库": {
            "主机": "localhost",
            "端口": 5432,
            "用户名": "admin"
        }
    }
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(测试配置, f, ensure_ascii=False, indent=2)
        临时文件路径 = f.name
    
    try:
        # 测试加载配置
        配置 = 配置管理器(临时文件路径)
        print(f"\n加载的配置: {配置}")
        
        # 测试获取配置项
        应用名称 = 配置.安全获取配置项('应用名称')
        print(f"应用名称: {应用名称}")
        
        # 测试嵌套键
        数据库主机 = 配置.安全获取配置项('数据库.主机')
        print(f"数据库主机: {数据库主机}")
        
        # 测试默认值
        不存在的键 = 配置.安全获取配置项('不存在的键', 默认值='默认值')
        print(f"不存在的键: {不存在的键}")
        
        # 测试验证配置
        验证结果 = 配置.验证配置完整性(['应用名称', '版本', '数据库.主机'])
        print(f"\n验证结果: {验证结果}")
        
        # 测试更新配置
        配置.更新配置('版本', '2.0.0')
        配置.更新配置('数据库.端口', 3306)
        print(f"\n更新后的版本: {配置.安全获取配置项('版本')}")
        print(f"更新后的端口: {配置.安全获取配置项('数据库.端口')}")
        
        # 测试字典式访问
        print(f"\n字典式访问: {配置['应用名称']}")
        配置['新配置项'] = '测试值'
        print(f"新配置项: {配置['新配置项']}")
        
    finally:
        # 清理临时文件
        os.unlink(临时文件路径)
    
    print("\n配置管理器测试完成")