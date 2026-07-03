"""
模型版本管理系统
提供模型版本的注册、查询、比较、标记和回滚功能
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class 模型版本管理器:
    """模型版本管理器类 - 管理所有模型版本的元数据和操作"""
    
    def __init__(self):
        self.当前目录 = Path(__file__).parent
        self.数据库路径 = self.当前目录 / "模型文件" / "模型版本数据库.json"
        self.模型目录 = self.当前目录 / "模型文件"
        self.确保数据库存在()
    
    def 确保数据库存在(self):
        """确保版本数据库文件存在"""
        if not self.数据库路径.exists():
            初始数据 = {
                "版本列表": [],
                "最佳版本": None,
                "创建时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.保存数据库(初始数据)
            print(f"创建版本数据库: {self.数据库路径}")
    
    def 加载数据库(self) -> Dict:
        """加载版本数据库"""
        try:
            if self.数据库路径.exists():
                with open(self.数据库路径, 'r', encoding='utf-8') as 文件:
                    return json.load(文件)
            else:
                return self.创建空数据库()
        except Exception as 错误:
            print(f"加载版本数据库失败: {错误}")
            return self.创建空数据库()
    
    def 创建空数据库(self) -> Dict:
        """创建空数据库结构"""
        return {
            "版本列表": [],
            "最佳版本": None,
            "创建时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def 保存数据库(self, 数据: Dict):
        """保存版本数据库"""
        数据["更新时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 确保目录存在
        self.数据库路径.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.数据库路径, 'w', encoding='utf-8') as 文件:
            json.dump(数据, 文件, ensure_ascii=False, indent=2)
    
    def 注册新版本(self, 模型路径: str, 训练参数: Dict = None, 
                    性能指标: Dict = None, 备注: str = "") -> bool:
        """注册新的模型版本
        
        参数:
            模型路径: 模型文件的路径（相对或绝对）
            训练参数: 训练使用的参数字典
            性能指标: 模型的性能指标（如准确率、损失值等）
            备注: 版本的备注说明
        
        返回:
            是否成功注册
        """
        try:
            模型路径对象 = Path(模型路径)
            
            # 如果是相对路径，转换为绝对路径
            if not 模型路径对象.is_absolute():
                模型路径对象 = self.当前目录 / 模型路径
            
            # 检查模型是否存在
            if not 模型路径对象.exists():
                print(f"模型路径不存在: {模型路径对象}")
                return False
            
            # 检查是否是有效的模型目录
            if not (模型路径对象 / "adapter_config.json").exists():
                print(f"不是有效的训练模型目录: {模型路径对象}")
                return False
            
            # 加载现有数据库
            数据库 = self.加载数据库()
            
            # 创建版本信息
            版本信息 = {
                "版本ID": self.生成版本ID(),
                "模型名称": 模型路径对象.name,
                "模型路径": str(模型路径对象.relative_to(self.当前目录)),  # 使用相对路径
                "注册时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "训练参数": 训练参数 or {},
                "性能指标": 性能指标 or {},
                "备注": 备注,
                "是否最佳": False,
                "文件大小": self.计算模型大小(模型路径对象),
                "训练样本数": 训练参数.get("样本数量", 0) if 训练参数 else 0
            }
            
            # 加载模型的详细信息（如果存在）
            模型信息文件 = 模型路径对象 / "模型信息.json"
            if 模型信息文件.exists():
                try:
                    with open(模型信息文件, 'r', encoding='utf-8') as f:
                        模型详情 = json.load(f)
                        版本信息["基础模型"] = 模型详情.get("基础模型", "未知")
                        版本信息["训练时间"] = 模型详情.get("训练时间", "未知")
                except:
                    pass
            
            # 添加到版本列表
            数据库["版本列表"].append(版本信息)
            
            # 保存数据库
            self.保存数据库(数据库)
            
            print(f"✓ 成功注册版本: {版本信息['版本ID']}")
            print(f"  模型名称: {版本信息['模型名称']}")
            print(f"  注册时间: {版本信息['注册时间']}")
            
            return True
            
        except Exception as 错误:
            print(f"注册版本失败: {错误}")
            return False
    
    def 生成版本ID(self) -> str:
        """生成唯一的版本ID"""
        时间戳 = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v{时间戳}"
    
    def 计算模型大小(self, 模型路径: Path) -> float:
        """计算模型文件大小（MB）"""
        try:
            总大小 = 0
            for 文件 in 模型路径.rglob("*"):
                if 文件.is_file():
                    总大小 += 文件.stat().st_size
            return round(总大小 / (1024 * 1024), 2)  # MB
        except:
            return 0.0
    
    def 列出所有版本(self) -> List[Dict]:
        """列出所有已注册的模型版本"""
        数据库 = self.加载数据库()
        版本列表 = 数据库.get("版本列表", [])
        
        if not 版本列表:
            print("\n没有找到任何已注册的模型版本")
            return []
        
        print("\n" + "=" * 70)
        print("已注册的模型版本列表")
        print("=" * 70)
        
        for 索引, 版本 in enumerate(版本列表, 1):
            最佳标记 = " [最佳]" if 版本.get("是否最佳", False) else ""
            print(f"\n{索引}. 版本ID: {版本['版本ID']}{最佳标记}")
            print(f"   模型名称: {版本['模型名称']}")
            print(f"   注册时间: {版本['注册时间']}")
            print(f"   训练时间: {版本.get('训练时间', '未知')}")
            print(f"   基础模型: {版本.get('基础模型', '未知')}")
            print(f"   文件大小: {版本['文件大小']} MB")
            print(f"   训练样本数: {版本.get('训练样本数', 0)}")
            
            if 版本.get('性能指标'):
                print(f"   性能指标:")
                for 指标名, 指标值 in 版本['性能指标'].items():
                    print(f"      {指标名}: {指标值}")
            
            if 版本.get('备注'):
                print(f"   备注: {版本['备注']}")
        
        print("\n" + "=" * 70)
        print(f"总计: {len(版本列表)} 个版本")
        
        return 版本列表
    
    def 比较版本(self, 版本ID1: str, 版本ID2: str) -> Dict:
        """比较两个版本的详细信息
        
        参数:
            版本ID1: 第一个版本的ID
            版本ID2: 第二个版本的ID
        
        返回:
            比较结果字典
        """
        数据库 = self.加载数据库()
        版本列表 = 数据库.get("版本列表", [])
        
        版本1 = None
        版本2 = None
        
        for 版本 in 版本列表:
            if 版本['版本ID'] == 版本ID1:
                版本1 = 版本
            if 版本['版本ID'] == 版本ID2:
                版本2 = 版本
        
        if not 版本1 or not 版本2:
            print(f"找不到指定的版本")
            return {}
        
        print("\n" + "=" * 70)
        print("版本比较结果")
        print("=" * 70)
        
        比较结果 = {
            "版本1": 版本1,
            "版本2": 版本2,
            "差异": {}
        }
        
        # 比较基本信息
        print(f"\n版本1: {版本1['版本ID']} - {版本1['模型名称']}")
        print(f"版本2: {版本2['版本ID']} - {版本2['模型名称']}")
        
        print("\n基本信息对比:")
        print(f"  注册时间: {版本1['注册时间']} vs {版本2['注册时间']}")
        print(f"  文件大小: {版本1['文件大小']} MB vs {版本2['文件大小']} MB")
        print(f"  训练样本数: {版本1.get('训练样本数', 0)} vs {版本2.get('训练样本数', 0)}")
        
        # 比较训练参数
        if 版本1.get('训练参数') and 版本2.get('训练参数'):
            print("\n训练参数对比:")
            参数1 = 版本1['训练参数']
            参数2 = 版本2['训练参数']
            
            常见参数 = ['学习率', '训练轮数', '批次大小', '梯度累积', '权重衰减']
            for 参数名 in 常见参数:
                值1 = 参数1.get(参数名, 'N/A')
                值2 = 参数2.get(参数名, 'N/A')
                差异标记 = " [不同]" if 值1 != 值2 else ""
                print(f"  {参数名}: {值1} vs {值2}{差异标记}")
        
        # 比较性能指标
        if 版本1.get('性能指标') and 版本2.get('性能指标'):
            print("\n性能指标对比:")
            指标1 = 版本1['性能指标']
            指标2 = 版本2['性能指标']
            
            所有指标名 = set(指标1.keys()) | set(指标2.keys())
            for 指标名 in 所有指标名:
                值1 = 指标1.get(指标名, 'N/A')
                值2 = 指标2.get(指标名, 'N/A')
                差异标记 = " [不同]" if 值1 != 值2 else ""
                print(f"  {指标名}: {值1} vs {值2}{差异标记}")
        
        print("\n" + "=" * 70)
        
        return 比较结果
    
    def 标记最佳版本(self, 版本ID: str) -> bool:
        """标记指定的版本为最佳版本
        
        参数:
            版本ID: 要标记的版本ID
        
        返回:
            是否成功标记
        """
        数据库 = self.加载数据库()
        版本列表 = 数据库.get("版本列表", [])
        
        找到版本 = False
        
        # 先清除之前的最佳标记
        for 版本 in 版本列表:
            if 版本['版本ID'] == 版本ID:
                版本['是否最佳'] = True
                找到版本 = True
            else:
                版本['是否最佳'] = False
        
        if 找到版本:
            数据库['最佳版本'] = 版本ID
            self.保存数据库(数据库)
            
            print(f"\n✓ 已标记版本 {版本ID} 为最佳版本")
            print("  其他版本的最佳标记已清除")
            return True
        else:
            print(f"\n找不到版本: {版本ID}")
            return False
    
    def 获取最佳版本(self) -> Optional[Dict]:
        """获取当前标记的最佳版本
        
        返回:
            最佳版本的信息字典，如果没有则返回None
        """
        数据库 = self.加载数据库()
        版本列表 = 数据库.get("版本列表", [])
        
        for 版本 in 版本列表:
            if 版本.get('是否最佳', False):
                return 版本
        
        return None
    
    def 回滚到版本(self, 版本ID: str) -> bool:
        """回滚到指定的模型版本（复制模型文件）
        
        参数:
            版本ID: 要回滚到的版本ID
        
        返回:
            是否成功回滚
        """
        数据库 = self.加载数据库()
        版本列表 = 数据库.get("版本列表", [])
        
        目标版本 = None
        for 版本 in 版本列表:
            if 版本['版本ID'] == 版本ID:
                目标版本 = 版本
                break
        
        if not 目标版本:
            print(f"找不到版本: {版本ID}")
            return False
        
        try:
            原路径 = self.当前目录 / 目标版本['模型路径']
            
            if not 原路径.exists():
                print(f"模型文件不存在: {原路径}")
                return False
            
            # 创建回滚目录名
            回滚目录名 = f"当前使用版本_回滚自_{版本ID}"
            目标路径 = self.模型目录 / 回滚目录名
            
            # 如果目标路径已存在，先删除
            if 目标路径.exists():
                print(f"目标路径已存在，正在删除: {目标路径}")
                shutil.rmtree(目标路径)
            
            # 复制模型文件
            print(f"\n正在回滚到版本 {版本ID}...")
            print(f"  从: {原路径}")
            print(f"  到: {目标路径}")
            
            shutil.copytree(原路径, 目标路径)
            
            # 创建回滚信息文件
            回滚信息 = {
                "回滚时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "源版本ID": 版本ID,
                "源版本名称": 目标版本['模型名称'],
                "源路径": str(原路径),
                "回滚路径": str(目标路径)
            }
            
            回滚信息文件 = 目标路径 / "回滚信息.json"
            with open(回滚信息文件, 'w', encoding='utf-8') as f:
                json.dump(回滚信息, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ 成功回滚到版本 {版本ID}")
            print(f"  回滚后的模型位置: {目标路径}")
            print(f"  回滚信息已记录在: {回滚信息文件}")
            
            return True
            
        except Exception as 错误:
            print(f"回滚失败: {错误}")
            return False
    
    def 删除版本记录(self, 版本ID: str, 删除文件: bool = False) -> bool:
        """删除版本记录（可选择是否删除实际文件）
        
        参数:
            版本ID: 要删除的版本ID
            删除文件: 是否同时删除模型文件
        
        返回:
            是否成功删除
        """
        数据库 = self.加载数据库()
        版本列表 = 数据库.get("版本列表", [])
        
        目标版本 = None
        新版本列表 = []
        
        for 版本 in 版本列表:
            if 版本['版本ID'] == 版本ID:
                目标版本 = 版本
            else:
                新版本列表.append(版本)
        
        if not 目标版本:
            print(f"找不到版本: {版本ID}")
            return False
        
        # 如果是最佳版本，清除最佳标记
        if 目标版本.get('是否最佳', False):
            数据库['最佳版本'] = None
        
        try:
            # 删除实际文件（如果选择）
            if 删除文件:
                模型路径 = self.当前目录 / 目标版本['模型路径']
                if 模型路径.exists():
                    print(f"\n正在删除模型文件: {模型路径}")
                    shutil.rmtree(模型路径)
                    print("  模型文件已删除")
            
            # 更新数据库
            数据库['版本列表'] = 新版本列表
            self.保存数据库(数据库)
            
            print(f"\n✓ 已删除版本记录: {版本ID}")
            if not 删除文件:
                print("  注意: 模型文件保留，仅删除版本记录")
            
            return True
            
        except Exception as 错误:
            print(f"删除失败: {错误}")
            return False
    
    def 添加性能指标(self, 版本ID: str, 性能指标: Dict) -> bool:
        """为指定版本添加性能指标
        
        参数:
            版本ID: 版本的ID
            性能指标: 性能指标字典
        
        返回:
            是否成功添加
        """
        数据库 = self.加载数据库()
        版本列表 = 数据库.get("版本列表", [])
        
        找到版本 = False
        
        for 版本 in 版本列表:
            if 版本['版本ID'] == 版本ID:
                # 合并性能指标
                if 版本.get('性能指标'):
                    版本['性能指标'].update(性能指标)
                else:
                    版本['性能指标'] = 性能指标
                找到版本 = True
                break
        
        if 找到版本:
            self.保存数据库(数据库)
            print(f"\n✓ 已为版本 {版本ID} 添加性能指标")
            for 指标名, 指标值 in 性能指标.items():
                print(f"  {指标名}: {指标值}")
            return True
        else:
            print(f"\n找不到版本: {版本ID}")
            return False
    
    def 自动扫描并注册模型(self) -> int:
        """自动扫描模型目录并注册所有未注册的模型
        
        返回:
            新注册的模型数量
        """
        print("\n正在扫描模型目录...")
        
        数据库 = self.加载数据库()
        已注册路径 = {版本['模型路径'] for 版本 in 数据库.get("版本列表", [])}
        
        新注册数量 = 0
        
        # 扫描模型目录
        for 模型路径 in self.模型目录.iterdir():
            if not 模型路径.is_dir():
                continue
            
            # 检查是否是有效的训练模型
            if not (模型路径 / "adapter_config.json").exists():
                continue
            
            # 计算相对路径
            相对路径 = str(模型路径.relative_to(self.当前目录))
            
            # 检查是否已注册
            if 相对路径 in 已注册路径:
                print(f"  跳过已注册模型: {模型路径.name}")
                continue
            
            # 加载模型信息
            模型信息文件 = 模型路径 / "模型信息.json"
            训练参数 = {}
            
            if 模型信息文件.exists():
                try:
                    with open(模型信息文件, 'r', encoding='utf-8') as f:
                        模型信息 = json.load(f)
                        训练参数 = 模型信息.get('训练参数', {})
                        训练参数['样本数量'] = 模型信息.get('样本数量', 0)
                except:
                    pass
            
            # 注册新版本
            if self.注册新版本(
                str(模型路径),
                训练参数=训练参数,
                备注="自动扫描注册"
            ):
                新注册数量 += 1
        
        print(f"\n✓ 扫描完成，新注册 {新注册数量} 个模型")
        return 新注册数量
    
    def 显示版本统计(self):
        """显示版本统计信息"""
        数据库 = self.加载数据库()
        版本列表 = 数据库.get("版本列表", [])
        
        if not 版本列表:
            print("\n没有找到任何模型版本")
            return
        
        print("\n" + "=" * 70)
        print("模型版本统计")
        print("=" * 70)
        
        总版本数 = len(版本列表)
        最佳版本 = self.获取最佳版本()
        总大小 = sum(版本.get('文件大小', 0) for 版本 in 版本列表)
        平均大小 = 总大小 / 总版本数 if 总版本数 > 0 else 0
        
        print(f"\n总版本数: {总版本数}")
        print(f"总文件大小: {round(总大小, 2)} MB")
        print(f"平均文件大小: {round(平均大小, 2)} MB")
        
        if 最佳版本:
            print(f"\n当前最佳版本: {最佳版本['版本ID']}")
            print(f"  模型名称: {最佳版本['模型名称']}")
            print(f"  注册时间: {最佳版本['注册时间']}")
        else:
            print("\n未设置最佳版本")
        
        # 按训练样本数排序
        有样本数的版本 = [v for v in 版本列表 if v.get('训练样本数', 0) > 0]
        if 有样本数的版本:
            最大样本数版本 = max(有样本数的版本, key=lambda x: x.get('训练样本数', 0))
            最小样本数版本 = min(有样本数的版本, key=lambda x: x.get('训练样本数', 0))
            
            print(f"\n训练样本数最多: {最大样本数版本['版本ID']} ({最大样本数版本.get('训练样本数', 0)} 样本)")
            print(f"训练样本数最少: {最小样本数版本['版本ID']} ({最小样本数版本.get('训练样本数', 0)} 样本)")
        
        print("\n" + "=" * 70)


# ==================== 便捷函数 ====================

def 注册新版本(模型路径: str, 训练参数: Dict = None, 
               性能指标: Dict = None, 备注: str = "") -> bool:
    """注册新模型版本的便捷函数"""
    管理器 = 模型版本管理器()
    return 管理器.注册新版本(模型路径, 训练参数, 性能指标, 备注)


def 列出所有版本() -> List[Dict]:
    """列出所有版本的便捷函数"""
    管理器 = 模型版本管理器()
    return 管理器.列出所有版本()


def 比较版本(版本ID1: str, 版本ID2: str) -> Dict:
    """比较两个版本的便捷函数"""
    管理器 = 模型版本管理器()
    return 管理器.比较版本(版本ID1, 版本ID2)


def 标记最佳版本(版本ID: str) -> bool:
    """标记最佳版本的便捷函数"""
    管理器 = 模型版本管理器()
    return 管理器.标记最佳版本(版本ID)


def 回滚到版本(版本ID: str) -> bool:
    """回滚到版本的便捷函数"""
    管理器 = 模型版本管理器()
    return 管理器.回滚到版本(版本ID)


def 自动扫描注册() -> int:
    """自动扫描并注册模型的便捷函数"""
    管理器 = 模型版本管理器()
    return 管理器.自动扫描并注册模型()


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("模型版本管理系统测试")
    print("=" * 60)
    
    管理器 = 模型版本管理器()
    
    # 自动扫描并注册模型
    管理器.自动扫描并注册模型()
    
    # 显示版本统计
    管理器.显示版本统计()
    
    # 列出所有版本
    管理器.列出所有版本()