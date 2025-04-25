from dataclasses import dataclass, field
from pathlib import Path
import sys
import os
import logging
import pandas as pd
import warnings
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()
logging.getLogger("huobi-client").setLevel(logging.WARNING)

# 忽略 pandas 的未來警告
warnings.filterwarnings("ignore", category=FutureWarning)


def get_project_root() -> Path:
    """假設 config.py 位於 src/bayes_exercise/config/，往上三層為 project 根目錄"""
    return Path(__file__).resolve().parents[3]


@dataclass
class Config:
    base_dir: Path = field(default_factory=get_project_root)
    data_dir: Path = field(default_factory=lambda: get_project_root() / "data")
    src_dir: Path = field(default_factory=lambda: get_project_root() / "src")
    reports_dir: Path = field(default_factory=lambda: get_project_root() / "reports")
    zero: float = sys.float_info.min
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", "demo_key"))
    secret_key: str = field(
        default_factory=lambda: os.getenv("SECRET_KEY", "demo_secret")
    )


def configure_pandas():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    pd.set_option("display.width", 300)


# 初始化全局配置對象
config = Config()
configure_pandas()

if __name__ == "__main__":
    print(f"Base Directory: {config.base_dir}")
    print(f"Data Directory: {config.data_dir}")
    print(f"Source Directory: {config.src_dir}")
    print(f"Reports Directory: {config.reports_dir}")
    print(f"API Key: {config.api_key}")
    print(f"Secret Key: {config.secret_key}")
