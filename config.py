from dataclasses import dataclass, field
import os
import sys
import pandas as pd

def default_base_dir() -> str:
    """返回项目根目录的路径。"""
    return os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    base_dir: str = field(default_factory=default_base_dir)
    data_dir: str = field(default_factory=lambda: os.path.join(default_base_dir(), 'data'))
    src_dir: str = field(default_factory=lambda: os.path.join(default_base_dir(), 'src'))
    config_dir: str = field(default_factory=lambda: os.path.join(default_base_dir(), 'conf'))
    reports_dir: str = field(default_factory=lambda: os.path.join(default_base_dir(), 'reports'))
    zero: float = sys.float_info.min

def configure_pandas():
    """配置 pandas 的显示选项。"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.width', 300)

# 创建 Config 的一个实例，这将是全局可用的
config = Config()

# 应用 pandas 配置
configure_pandas()

# 使用例子
if __name__ == "__main__":
    print(f"Base Directory: {config.base_dir}")
    print(f"Data Directory: {config.data_dir}")
    print(f"Source Directory: {config.src_dir}")
