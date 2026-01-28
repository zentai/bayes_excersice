import pandas as pd
from pathlib import Path
from quanthunt.hunterverse.interface import IMarketSensor, StrategyParam, DEBUG_COL
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
    def __init__(self, symbol, interval):
        super().__init__(symbol, interval)
        self.last_published = None

    def scan(self, limits):
        df = pandas_util.load_symbols_from_huobi(self.symbol, limits, self.interval)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        self.last_published = df.iloc[-2].Date
        return df[:-1]  # because the last raw still process

    def fetch_one(self):
        new_data = pandas_util.get_history_stick(
            self.symbol, sample=3, interval=self.interval
        )
        columns = new_data.columns
        new_data = new_data.iloc[1]
        new_data["Matured"] = pd.NaT
        if self.last_published == new_data.Date:
            return pd.DataFrame([], columns=columns)
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


class StateMarketSensor(IMarketSensor):
    """
    State-aware incremental market sensor.

    Compare:
    - report/<symbol>_<interval>.csv  (strategy-known world)
    - data/<symbol>_<interval>.csv    (market truth)

    Return only unseen rows (by Date).
    """

    def __init__(self, symbol: str, interval: str, sp: StrategyParam):
        super().__init__(symbol, interval)

        self.sp = sp

        # === directories from config ===
        self.data_dir = Path(config.data_dir)
        self.report_dir = Path(config.reports_dir)

        self.data_path = self.data_dir / f"{symbol}_{interval}.csv"
        self.report_path = self.report_dir / f"{sp}.csv"

        self._new_df: pd.DataFrame | None = None

    # -------------------------------------------------
    # Public API (align with LocalMarketSensor)
    # -------------------------------------------------

    def scan(self) -> pd.DataFrame:
        """
        Load report as base_df (already calculated world).
        """
        if not self.report_path.exists():
            self._prepare_diff()
            return self._new_df

        report_df = pd.read_csv(self.report_path)
        report_df["Date"] = pd.to_datetime(report_df["Date"], errors="coerce")
        report_df = report_df.dropna(subset=["Date"])
        report_df = report_df.sort_values("Date")
        return report_df

    def fetch(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge unseen market data into base_df.

        Rules:
        - If report/base_df is empty → directly return new data
        - If no new data → return base_df unchanged
        - Otherwise → append new data to base_df
        """
        self._prepare_diff(base_df)

        # 沒有新資料，世界不前進
        if self._new_df.empty:
            return base_df

        # cold start：report 原本就沒有東西
        if base_df.empty:
            return self._new_df.copy()

        # 正常情況：append 新資料
        return pd.concat([base_df, self._new_df], ignore_index=True)

    def left(self) -> int:
        if self._new_df is None:
            self._prepare_diff()

        return len(self._new_df)

    # -------------------------------------------------
    # Internal
    # -------------------------------------------------

    def _prepare_diff(self, report_df=pd.DataFrame()):
        """
        Compute data - report difference by Date.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"[StateMarketSensor] data file not found: {self.data_path}"
            )

        # --- load market truth ---
        data_df = pd.read_csv(self.data_path)
        data_df["Date"] = pd.to_datetime(data_df["Date"], errors="coerce")
        data_df = data_df.dropna(subset=["Date"])
        data_df = data_df.sort_values("Date")

        # --- load strategy-known world ---
        if report_df.empty:
            self._new_df = data_df.reset_index(drop=True)
            return

        report_df["Date"] = pd.to_datetime(report_df["Date"], errors="coerce")
        report_df = report_df.dropna(subset=["Date"])
        report_df = report_df.sort_values("Date")

        last_seen_date = report_df["Date"].max()
        max_data_date = data_df["Date"].max()

        # --- hard guard ---
        if last_seen_date > max_data_date:
            raise RuntimeError(
                f"[StateMarketSensor] report ahead of data: "
                f"report={last_seen_date}, data={max_data_date}"
            )

        # --- causal diff ---
        new_df = data_df[data_df["Date"] > last_seen_date]
        print("*" * 100)
        print(new_df)
        print("*" * 100)
        self._new_df = new_df.reset_index(drop=True)
