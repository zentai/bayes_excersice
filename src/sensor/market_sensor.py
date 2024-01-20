from hunterverse.interface import IMarketSensor
from utils import pandas_util
import pandas as pd


class LocalMarketSensor(IMarketSensor):
    def __init__(self, symbol, interval):
        super().__init__(symbol, interval)
        self.update_idx = 0
        self.test_df = None
        self.interval_sec = 0.01

    def scan(self, limits):
        df = pandas_util.load_symbols(self.symbol)
        self.test_df = df[limits:]
        df = df[:limits]
        return df

    def fetch(self, base_df):
        new_data = self.test_df.iloc[self.update_idx].copy()
        new_data["Matured"] = pd.NaT
        base_df = pd.concat(
            [base_df, pd.DataFrame([new_data], columns=base_df.columns)],
            ignore_index=True,
        )
        self.update_idx += 1
        return base_df


class HuobiMarketSensor(IMarketSensor):
    def scan(self, limits):
        df = pandas_util.load_symbols_from_huobi(self.symbol, limits, self.interval)
        return df

    def fetch(self, base_df):
        new_data = pandas_util.get_history_stick(
            self.symbol, sample=3, interval=self.interval
        )
        print(new_data)
        new_data = new_data.iloc[1]
        new_data["Matured"] = pd.NaT
        base_df = pd.concat(
            [base_df, pd.DataFrame([new_data], columns=base_df.columns)],
            ignore_index=True,
        )
        return base_df
