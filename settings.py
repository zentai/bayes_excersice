# env.py
import os
import sys
import pandas as pd
import logging

logging.getLogger("huobi-client").setLevel(logging.CRITICAL)

# 獲取當前文件的絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 定義其他文件夾和文件的絕對路徑
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")
CONFIG_DIR = os.path.join(BASE_DIR, "conf")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.width", 300)

ZERO = sys.float_info.min
