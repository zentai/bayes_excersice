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
from hunterverse.interface import INTERVAL_TO_MIN

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


def hunterPause(sp):
    if sp.fetch_huobi:
        now = datetime.datetime.now()
        minutes_interval = INTERVAL_TO_MIN.get(sp.interval)
        next_whole_point = now.replace(second=0, microsecond=0) + datetime.timedelta(
            minutes=minutes_interval
        )
        if now.minute % minutes_interval != 0:
            next_whole_point = next_whole_point.replace(
                minute=(now.minute // minutes_interval + 1) * minutes_interval
            )
        sleep_seconds = (next_whole_point - now).total_seconds()
        seconds_to_wait = max(0, sleep_seconds) + 5  # Ensure non-negative value
        print(
            f"Will be start after: {seconds_to_wait} sec, {datetime.datetime.now()+datetime.timedelta(seconds=seconds_to_wait)}"
        )
        time.sleep(seconds_to_wait)


def start_journey(sp):
    if sp.simulate:
        sensor = LocalMarketSensor(symbol=sp.symbol, interval="local")
    elif sp.fetch_huobi:
        sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)

    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)

    # Adjust start time
    hunterPause(sp)

    story = HuntingStory(sensor, scout, engine, hunter)
    base_df = sensor.scan(2000)
    final_review = None

    round = sensor.left() or 1000000
    for i in range(round):
        try:
            base_df, review = story.move_forward(base_df)
            final_review = review
            print(base_df[DEBUG_COL][-30:])
            print(final_review)
            base_df[DUMP_COL].to_csv(f"{REPORTS_DIR}/{sp.symbol.name}.csv", index=False)
            print(f"{REPORTS_DIR}/{sp.symbol.name}.csv")
            hunterPause(sp)

        except Exception as e:
            print(e)
            time.sleep(5)
    return final_review


def training_camp(sp):
    sensor = LocalMarketSensor(symbol=sp.symbol, interval="local")
    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)

    story = HuntingStory(sensor, scout, engine, hunter)
    base_df = sensor.scan(1000)
    final_review = None

    for i in range(sensor.left()):
        base_df, review = story.move_forward(base_df)
        final_review = review
    return base_df, final_review


@click.command()
@click.option("--ccy", default="maticusdt", required=False, help="trade ccy pair")
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
def main(ccy, interval, fund):
    params = {
        "ATR_sample": 40.69386655044119,
        "atr_loss_margin": 1.0,
        "bayes_windows": 69.13338800480899,
        "lower_sample": 100.0,
        "upper_sample": 5.0,
        "interval": "15min",
        "symbol": Symbol(ccy),
        "fetch_huobi": True,
        "simulate": False,
    }
    sp = StrategyParam(**params)
    final_review = training_camp(sp)


if __name__ == "__main__":
    main()
