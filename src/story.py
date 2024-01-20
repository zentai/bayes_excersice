import sys
import os
import time
import click
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataclasses import dataclass
import pandas as pd
import numpy as np
from settings import DATA_DIR, SRC_DIR, REPORTS_DIR

from strategy.turtle_trading import TurtleScout
from engine.probabilistic_engine import BayesianEngine

from hunterverse.interface import IStrategyScout
from hunterverse.interface import IMarketSensor
from hunterverse.interface import IEngine
from hunterverse.interface import IHunter
from hunterverse.interface import Symbol
from hunterverse.interface import StrategyParam

from sensor.market_sensor import LocalMarketSensor
from sensor.market_sensor import HuobiMarketSensor
from tradingfirm.trader import xHunter


DEBUG_COL = [
    "Date",
    # "Open",
    # "High",
    "Low",
    "Close",
    # "BuySignal",
    # "Stop_profit",
    "exit_price",
    # "Matured",
    # "time_cost",
    "buy",
    "sell",
    "profit",
    "xBuy",
    "xSell",
    "xProfit",
    "xPosition",
    "xCash",
    "xAvgCost",
    "Kelly",
    "Postrior",
    "P/L",
    "likelihood",
    "profit_margin",
    "loss_margin",
]

DUMP_COL = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "BuySignal",
    "Stop_profit",
    "exit_price",
    "Matured",
    "time_cost",
    "buy",
    "sell",
    "profit",
    # "turtle_l",
    # "turtle_h",
    "xBuy",
    "xSell",
    "xProfit",
    "xPosition",
    "xCash",
    "xAvgCost",
    "Kelly",
    "Postrior",
    "P/L",
    "likelihood",
    "profit_margin",
    "loss_margin",
]


@dataclass
class HuntingStory:
    sensor: IMarketSensor
    scout: IStrategyScout
    engine: IEngine
    hunter: IHunter

    def move_forward(self, base_df):
        base_df = self.scout.market_recon(base_df)
        base_df = self.engine.hunt_plan(base_df)
        base_df = self.hunter.strike_phase(base_df)
        base_df = self.sensor.fetch(base_df)
        return base_df, self.hunter.review_mission(base_df)


def start_journey(sp):
    if sp.simulate:
        sensor = LocalMarketSensor(symbol=sp.symbol, interval="local")
    elif sp.fetch_huobi:
        sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)

    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)

    # Adjust start time
    if sp.fetch_huobi:
        seconds_to_wait = 60 - datetime.datetime.now().second + 5
        print(f"Will be start after: {seconds_to_wait} sec")
        time.sleep(seconds_to_wait)

    story = HuntingStory(sensor, scout, engine, hunter)
    base_df = sensor.scan(1000)
    final_review = None

    round = sensor.left() or 1000000
    for i in range(round):
        # base_df, review = story.move_forward(base_df)
        try:
            base_df, review = story.move_forward(base_df)
            print(base_df[DEBUG_COL][-30:])
            print(final_review)
            base_df[DUMP_COL].to_csv(f"{REPORTS_DIR}/{sp.symbol.name}.csv", index=False)
            print(f"{REPORTS_DIR}/{sp.symbol.name}.csv")

            if sp.fetch_huobi:
                seconds_to_wait = 60 - datetime.datetime.now().second + 5
                print(f"Will be start after: {seconds_to_wait} sec")
                time.sleep(seconds_to_wait)
        except Exception as e:
            print(e)
            time.sleep(5)
        final_review = review
    return final_review


@click.command()
@click.option("--ccy", required=False, help="trade ccy pair")
@click.option(
    "--interval",
    required=False,
    help="trade interval: 1min 5min 15min 30min 60min 4hour 1day 1mon 1week 1year",
)
@click.option(
    "--fund",
    required=False,
    type=int,
    help="initial funds",
)
def main(ccy="maticusdt", interval=None, fund=None):
    params = {
        "ATR_sample": 73,
        "atr_loss_margin": 10,
        "bayes_windows": 5,
        "lower_sample": 13,
        "upper_sample": 5,
        "interval": "1min",
        "symbol": Symbol(ccy),
        "fetch_huobi": True,
        "simulate": False,
    }
    sp = StrategyParam(**params)
    final_review = start_journey(sp)


if __name__ == "__main__":
    main()
