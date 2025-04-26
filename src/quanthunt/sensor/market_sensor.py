import pandas as pd
from quanthunt.hunterverse.interface import IMarketSensor
from quanthunt.utils import pandas_util
import yfinance as yf
import datetime
from datetime import timedelta
from quanthunt.config.core_config import config

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir
import os


class YahooMarketSensor(IMarketSensor):
    def __init__(self, symbol, interval):
        super().__init__(symbol, interval)
        self.symbol = symbol
        self.update_idx = 0
        self.test_df = None
        self.interval_sec = 0.01

    def scan(self, limits):
        path = config.data_dir / f"{self.symbol}_cached.csv"
        if path.exists():
            print(f"Loading cached file: {path}")
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            df = df.sort_values("Date")
            df = df[-limits:]
        else:
            today = datetime.date.today()
            start_date = today - timedelta(days=limits)
            print(f"Downloading from Yahoo: {start_date} to {today}")
            df = yf.download(self.symbol.name, start=start_date, end=today)
            df = df.rename(columns={"Volume": "Vol"})
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            df = df.sort_values("Date")
        self.test_df = df.copy()
        return df

    def fetch_one(self):
        if self.test_df is not None and self.update_idx < len(self.test_df):
            row = self.test_df.iloc[self.update_idx]
            self.update_idx += 1
            return pd.DataFrame([row], columns=self.test_df.columns)
        else:
            return pd.DataFrame()

    def fetch(self, base_df):
        new_data = self.fetch_one()
        if not new_data.empty:
            base_df = pd.concat([base_df, new_data], ignore_index=True)
        return base_df

    def left(self):
        if self.test_df is not None:
            return len(self.test_df) - self.update_idx
        return 0


class LocalMarketSensor(IMarketSensor):
    def __init__(self, symbol, interval):
        super().__init__(symbol, interval)
        self.update_idx = 0
        self.test_df = None
        self.interval_sec = 0.01

    def scan(self, limits):
        df = pandas_util.load_symbols(self.symbol)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date")
        length = len(df)
        limits = min(length, limits)
        self.test_df = df[limits:]
        df = df[:limits]
        return df

    def fetch_one(self):
        new_data = (
            self.test_df.iloc[self.update_idx].copy()
            if not self.test_df.empty
            else self.test_df
        )
        new_data["Matured"] = pd.NaT
        self.update_idx += 1
        df = pd.DataFrame([new_data], columns=self.test_df.columns)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def fetch(self, base_df):
        new_data = self.fetch_one()
        base_df = pd.concat(
            [base_df, pd.DataFrame([new_data], columns=base_df.columns)],
            ignore_index=True,
        )
        return base_df

    def left(self):
        return len(self.test_df) - (self.update_idx + 1)


class HuobiMarketSensor(IMarketSensor):
    def scan(self, limits):
        df = pandas_util.load_symbols_from_huobi(self.symbol, limits, self.interval)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        return df[:-2]

    def fetch_one(self):
        new_data = pandas_util.get_history_stick(
            self.symbol, sample=3, interval=self.interval
        )
        columns = new_data.columns
        new_data = new_data.iloc[1]
        new_data["Matured"] = pd.NaT
        return pd.DataFrame([new_data], columns=columns)

    def fetch(self, base_df):
        new_data = self.fetch_one()
        base_df = pd.concat(
            [base_df, new_data],
            ignore_index=True,
        )
        return base_df

    def left(self):
        return 1
