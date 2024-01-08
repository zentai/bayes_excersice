import pandas as pd
from settings import DATA_DIR, SRC_DIR, REPORTS_DIR
import time
import huobi
from huobi.client.trade import TradeClient
from huobi.client.account import AccountClient
from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.utils import *
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

INTERVAL_TO_MIN = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 1 * 60,
    "4hour": 4 * 60,
    "1day": 24 * 60,
    "local": 0.001,
}


@dataclass
class Candlestick:
    Date: int
    High: float
    Low: float
    Open: float
    Close: float
    Amount: float
    Count: int
    Vol: float

    def __post_init__(self):
        self.Date = datetime.fromtimestamp(self.Date)


g_api_key = "uymylwhfeg-eb0f1107-98ea054c-ad39b"
g_secret_key = "6fc09731-0b3fde88-b83e1f10-2dc2f"

huobi_interval = {
    "1min": CandlestickInterval.MIN1,
    "5min": CandlestickInterval.MIN5,
    "15min": CandlestickInterval.MIN15,
    "30min": CandlestickInterval.MIN30,
    "60min": CandlestickInterval.MIN60,
    "4hour": CandlestickInterval.HOUR4,
    "1day": CandlestickInterval.DAY1,
    "1mon": CandlestickInterval.MON1,
    "1week": CandlestickInterval.WEEK1,
    "1year": CandlestickInterval.YEAR1,
}


def get_history_stick(symbol, sample=20, interval="1min"):
    interval = huobi_interval.get(interval)
    market_client = MarketClient(init_log=True, timeout=10)
    htx_stick = market_client.get_candlestick(symbol.name, interval, sample)

    candlesticks = [
        Candlestick(
            stick.id,
            stick.high,
            stick.low,
            stick.open,
            stick.close,
            stick.amount,
            stick.count,
            stick.vol,
        )
        for stick in htx_stick
    ]
    df = pd.DataFrame(candlesticks)
    return df.sort_values(by=["Date"]).reset_index(drop=True)


def load_symbols(symbol):
    code = symbol.name  # "BTC-USD"
    df = pd.read_csv(f"{DATA_DIR}/{code}.csv")
    df = df.dropna()
    return df


def load_symbols_from_huobi(symbol, limits, interval):
    df = get_history_stick(symbol, sample=limits, interval=interval)
    return df


def equip_fields(df, columns):
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        new_cols = {
            col: pd.NaT if col == "Matured" else np.nan for col in missing_columns
        }
        return df.assign(**new_cols)
    return df


# TODO: refactor me to nicer place
def sim_trade(symbol, action):
    market_client = MarketClient()
    seeking = True
    while seeking:
        list_obj = market_client.get_market_trade(symbol=symbol.name)
        for t in list_obj:
            if t.direction == "sell" if action == "buy" else "buy":
                return t.price
            # t.print_object()
        time.sleep(0.05)
