# hunterverse/storage.py

import os
import pandas as pd
from config import config


class HuntingCamp:
    def __init__(self, strategy_param):
        self.strategy_param = strategy_param
        self.file_path = f"{config.reports_dir}/{strategy_param}.csv"

    def load(self):
        if os.path.exists(self.file_path):
            print(self.file_path)
            df = pd.read_csv(self.file_path)
            print(f"Loaded DataFrame from {self.file_path}")
            return df
        else:
            print("No existing DataFrame file found, initializing new sensor data.")
            return pd.DataFrame()

    def save(self, df):
        df.to_csv(self.file_path, index=False)
        print(f"Saved DataFrame to {self.file_path}")
