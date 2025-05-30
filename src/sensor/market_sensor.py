import pandas as pd
import pymongo
from ..hunterverse.interface import IMarketSensor
from ..utils import pandas_util
import yfinance as yf
import datetime
from datetime import timedelta
from config import config

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
        if os.path.exists(f"{DATA_DIR}/{self.symbol}_cached.csv"):
            print(f"Load from file system: '{DATA_DIR}/{self.symbol}_cached.csv'")
            df = pd.read_csv(
                f"{DATA_DIR}/{self.symbol}_cached.csv",
                index_col="Date",
                parse_dates=True,
            )
            df = df[-limits:]
        else:
            today = datetime.datetime.today().date()
            start_date = today - timedelta(days=limits)
            print(f"load from yahoo finance: {start_date} {today}")
            data = yf.download(self.symbol.name, start=str(start_date), end=str(today))
            df = data.rename(columns={"Volume": "Vol"})
        return df

    def fetch(self, base_df):
        today = datetime.datetime.today().date()
        last_date = base_df.index[-1].date()
        data = yf.download(self.symbol.name, start=str(last_date), end=str(today))
        data = data.rename(columns={"Volume": "Vol"})
        data.index = data.index.map(lambda x: pd.Timestamp(x))
        base_df = pd.concat(
            [base_df, data],
            ignore_index=False,
        )
        base_df = base_df[~base_df.index.duplicated(keep="last")]
        return base_df


class LocalMarketSensor(IMarketSensor):
    def __init__(self, symbol, interval):
        super().__init__(symbol, interval)
        self.update_idx = 0
        self.test_df = None
        self.interval_sec = 0.01

    def scan(self, limits):
        df = pandas_util.load_symbols(self.symbol)
        df["Date"] = pd.to_datetime(df["Date"])
        length = len(df)
        limits = 0 if length < limits else limits
        self.test_df = df[limits:]
        df = df[:limits] if limits else df
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


class MongoDBHandler:
    def __init__(self, db_name="quant_database"):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]

    def save(self, collection_name, df):
        collection = self.db[collection_name]
        records = df.to_dict("records")
        collection.insert_many(records)
        print(f"Data saved to {self.db.name}.{collection.name} collection.")

    def load(self, collection_name):
        collection = self.db[collection_name]
        data = list(collection.find({}, {"_id": 0}))
        df = pd.DataFrame(data)
        print(f"Data loaded from {self.db.name}.{collection.name} collection.")
        return df


class MongoMarketSensor(IMarketSensor):
    def __init__(self, symbol, interval):
        super().__init__(symbol, interval)
        self.update_idx = 0
        self.test_df = None
        self.interval_sec = 0.01
        self.db = MongoDBHandler()

    def scan(self, limits):
        df = self.db.load(collection_name=f"{self.symbol.name}_raw")
        length = len(df)
        limits = 0 if length < limits else limits
        self.test_df = df[limits:]
        df = df[:limits] if limits else df
        return df

    def fetch_one(self):
        new_data = (
            self.test_df.iloc[self.update_idx].copy()
            if not self.test_df.empty
            else self.test_df
        )
        new_data["Matured"] = pd.NaT
        self.update_idx += 1
        return pd.DataFrame([new_data], columns=self.test_df.columns)

    def fetch(self, base_df):
        new_data = self.fetch_one()
        base_df = pd.concat(
            [base_df, pd.DataFrame([new_data], columns=base_df.columns)],
            ignore_index=True,
        )
        return base_df

    def left(self):
        return len(self.test_df) - (self.update_idx + 1)
