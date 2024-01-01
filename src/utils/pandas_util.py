from settings import DATA_DIR, SRC_DIR, REPORTS_DIR
import pandas as pd

import huobi
from huobi.client.trade import TradeClient
from huobi.client.account import AccountClient
from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.utils import *
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


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

def get_history_stick(symbol, sample=20, interval="1min"):
    symbol = symbol.replace('-', '').lower()
    print(symbol)
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

    interval = huobi_interval.get(interval)
    market_client = MarketClient(init_log=True, timeout=10)
    htx_stick = market_client.get_candlestick(symbol, interval, sample)

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


def load_symbols(symbols):
    code = symbols  # "BTC-USD"
    df = pd.read_csv(f"{DATA_DIR}/{code}.csv")
    df = df.dropna()
    return df

def load_symbols_from_huobi(symbols, interval):
    df = get_history_stick(symbols, sample=2000, interval=interval)
    return df
