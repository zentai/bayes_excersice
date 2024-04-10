from icecream import ic

import signal
import sys
import time
import click
import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np

from config import config
from huobi.constant.definition import OrderType
from ..tradingfirm.platforms import huobi_api

from ..strategy.turtle_trading import TurtleScout
from ..engine.probabilistic_engine import BayesianEngine

from ..hunterverse.interface import IStrategyScout
from ..hunterverse.interface import IMarketSensor
from ..hunterverse.interface import IEngine
from ..hunterverse.interface import IHunter
from ..hunterverse.interface import Symbol
from ..hunterverse.interface import StrategyParam
from ..hunterverse.interface import INTERVAL_TO_MIN

from ..sensor.market_sensor import LocalMarketSensor
from ..sensor.market_sensor import HuobiMarketSensor
from ..tradingfirm.trader import xHunter
from dataclasses import dataclass


@dataclass
class PortfolioItem:
    Pair: str
    MarketValue: float
    BuyValue: float
    ProfitLoss: float
    ProfitLossPercent: float
    Hr24: float
    Position: float
    AvgCost: float
    Strike: float
    Cash: float


def load_history(symbols, sp):
    hunter = xHunter(params=sp)
    hunter.load_memories()


def summarize(df):
    summary = {
        "Pair": "Summary",
        "MarketValue": df["MarketValue"].sum(),
        "BuyValue": df["BuyValue"].sum(),
        "ProfitLoss": df["ProfitLoss"].sum(),
        "ProfitLossPercent": df["ProfitLoss"].sum() / df["BuyValue"].sum(),
        "Hr24": df["Hr24"].mean(),
    }

    # 调整盈亏百分比和24小时变动为百分比形式
    summary["ProfitLossPercent"] = summary["ProfitLossPercent"] * 100
    summary["Hr24"] = summary["Hr24"] * 100

    # 将总结行添加到DataFrame
    df_summary = pd.DataFrame([summary], columns=df.columns)
    df_final = pd.concat([df, df_summary], ignore_index=True)
    return df_final


@click.command()
@click.option(
    "--portfolio", default="portfolio.csv", required=False, help="load protfolio"
)
def main(portfolio):
    portfolio_path = f"{config.data_dir}/{portfolio}"
    portfolio_df = pd.read_csv(portfolio_path)
    items = []
    for portfolio in portfolio_df.itertuples():
        sp = StrategyParam(
            **{
                "ATR_sample": 40.69386655044119,
                "atr_loss_margin": 1.0,
                "hard_cutoff": 0.95,
                "bayes_windows": 69.13338800480899,
                "lower_sample": 100.0,
                "upper_sample": 5.0,
                "funds": float(portfolio.fund),
                "interval": portfolio.interval,
                "symbol": Symbol(portfolio.ccy),
                "fetch_huobi": True,
                "simulate": False,
            }
        )
        history = huobi_api.get_history_stick(
            portfolio.ccy, sample=24, interval="60min"
        )
        hunter = xHunter(params=sp)
        hunter.load_memories()
        # hunter.load_memories(deals="".split(","))
        first_close, last_close = history.iloc[0]["Close"], history.iloc[-1]["Close"]
        items.append(hunter.portfolio(first_close, last_close))
        # break

    df = pd.DataFrame(items)
    df = summarize(df)
    print(df)


if __name__ == "__main__":
    main()
