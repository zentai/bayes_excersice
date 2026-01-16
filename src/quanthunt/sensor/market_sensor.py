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
        self.interval = interval
        self.interval_sec = 0.01

    def scan(self, limits=None):
        path = config.data_dir / f"{self.symbol}_{self.interval}.csv"
        today = datetime.date.today()

        if path.exists():
            print(f"Loading cached: {path}")
            df_cached = pd.read_csv(path, parse_dates=["Date"])
            df_cached = df_cached.sort_values("Date")

            last_date = df_cached["Date"].iloc[-1].date()
            print(f"Cached latest date: {last_date}")

            # 判斷是否需要更新
            if last_date < today:
                start_date = last_date + datetime.timedelta(days=1)
                print(f"Downloading incremental: {start_date} → {today}")

                df_new = yf.download(
                    self.symbol.name,
                    start=start_date,
                    end=today,
                    multi_level_index=False,
                )

                if not df_new.empty:
                    df_new = df_new.rename(columns={"Volume": "Vol"})
                    df_new = df_new.reset_index()
                    df_new["Date"] = pd.to_datetime(df_new["Date"])

                    # 合併舊+新
                    df = pd.concat([df_cached, df_new], ignore_index=True)
                    df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
                    # 立即寫回 cache
                    df.to_csv(path, index=False)
                else:
                    print("No new data from Yahoo, using cached data.")
                    df = df_cached
            else:
                print("Cached already up-to-date.")
                df = df_cached

        else:
            # 完全沒有 cache → Full download
            start_date = today - datetime.timedelta(days=2000)
            print(f"Downloading first time: {start_date} → {today}")

            df = yf.download(
                self.symbol.name, start=start_date, end=today, multi_level_index=False
            )
            df = df.rename(columns={"Volume": "Vol"}).reset_index()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

            # 寫入 cache
            df.to_csv(path, index=False)

        # 若需要，保留最近 limits 行
        if limits:
            df = df[-limits:]

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
        return base_df.sort_values("Date")

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
        self.interval = interval

    def scan(self, limits):
        df = pandas_util.load_symbols(self.symbol, self.interval)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date")
        length = int(len(df) / 2)
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
        return new_data

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
