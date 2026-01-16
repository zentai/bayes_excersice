import time
import json
import os
from pathlib import Path

from huobi.client.trade import TradeClient
from huobi.client.account import AccountClient
from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.utils import *
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np


from quanthunt.hunterverse.interface import Symbol, StrategyParam
from quanthunt.config.core_config import config

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir

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


def load_symbols(symbol, interval):
    code = symbol.name  # "BTC-USD"
    df = pd.read_csv(f"{DATA_DIR}/{code}_{interval}.csv")
    # df = df.dropna()
    return df[["Date", "Open", "High", "Low", "Close", "Vol"]]


def load_symbols_from_huobi(symbol, limits, interval):
    df = get_history_stick(symbol, sample=limits, interval=interval)
    return df


def equip_fields(df, columns):
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        new_cols = {
            col: (pd.NaT if col == "Matured" else np.nan) for col in missing_columns
        }
        missing_df = pd.DataFrame(new_cols, index=df.index)
        return pd.concat([df, missing_df], axis=1)
    return df


def buy_market(symbol, budget):
    trade_client = TradeClient(api_key=g_api_key, secret_key=g_secret_key)
    order_id = trade_client.create_order(
        symbol=symbol,
        account_id=account_id,
        order_type=OrderType.BUY_MARKET,
        source=OrderSource.API,
        amount=5.0,
        price=1.292,
    )


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


def setup_scheduler():
    import pytz
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.executors.pool import ProcessPoolExecutor
    from apscheduler.triggers.cron import CronTrigger

    jobstores = {}

    executors = {
        "default": {"type": "threadpool", "max_workers": 20},
        "processpool": ProcessPoolExecutor(max_workers=5),
    }

    job_defaults = {"coalesce": False, "max_instances": 10}

    scheduler = BackgroundScheduler()
    scheduler.configure(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=pytz.timezone("Asia/Singapore"),
    )

    return scheduler


def to_hz(interval: str, value: float) -> float:
    minutes_interval = INTERVAL_TO_MIN.get(interval)
    return value / (minutes_interval * 60)


def build_strategy_param(overrides: dict = {}) -> StrategyParam:
    default = {
        "ATR_sample": 60,
        "bayes_windows": 20,
        "lower_sample": 60,
        "upper_sample": 60,
        "hard_cutoff": 0.975,
        "profit_loss_ratio": 3.0,
        "atr_loss_margin": 1.0,
        "surfing_level": 5,
        "interval": "5min",
        "funds": 15,
        "stake_cap": 15,
        "symbol": Symbol("btcusdt"),
        "hmm_split": 4,
        "hmm_model": "trend",
        "backtest": False,
        "debug_mode": [
            "statement",
            "statement_to_csv",
            "mission_review",
            "final_statement_to_csv",
        ],
        "load_deals": [],
        "start_deal": "",
        "api_key": None,
        "secret_key": None,
    }
    default.update(overrides)
    return StrategyParam(**default)


def write_status(sp: StrategyParam, review_df: pd.DataFrame, status: str = "finished"):
    if review_df.empty:
        raise ValueError("Empty review dataframe, cannot write status.")

    result = review_df.iloc[0].to_dict()
    result = {k: (None if pd.isna(v) else v) for k, v in result.items()}
    result["symbol"] = sp.symbol.name
    status_data = {
        "task_id": f"{sp}",
        "status": status,
        "last_update": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": os.getpid(),
        "result": result,
    }

    status_path = Path(f"{config.data_dir}/status/{sp}.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w") as f:
        json.dump(status_data, f, indent=2)
