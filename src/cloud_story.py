import signal
import sys
import threading
import time
import click
import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np

from config import config
from huobi.constant.definition import OrderType
from .tradingfirm.platforms import huobi_api

from .strategy.turtle_trading import TurtleScout
from .engine.probabilistic_engine import BayesianEngine

from .hunterverse.interface import IStrategyScout
from .hunterverse.interface import IMarketSensor
from .hunterverse.interface import IEngine
from .hunterverse.interface import IHunter
from .hunterverse.interface import Symbol
from .hunterverse.interface import StrategyParam
from .hunterverse.interface import INTERVAL_TO_MIN

from .sensor.market_sensor import LocalMarketSensor
from .sensor.market_sensor import HuobiMarketSensor
from .tradingfirm.trader import xHunter
from pydispatch import dispatcher

DEBUG_COL = [
    "Date",
    # "Open",
    # "High",
    "Low",
    "Close",
    # "BuySignal",
    "Stop_profit",
    "exit_price",
    # "Matured",
    # "time_cost",
    "buy",
    "sell",
    "profit",
    "sBuy",
    "sSell",
    "sProfit",
    "sPosition",
    "sCash",
    "sAvgCost",
    "xBuy",
    "xSell",
    "xProfit",
    "xPosition",
    "xCash",
    "xAvgCost",
    "xBuyOrder",
    "xSellOrder",
    # "Kelly",
    # "Postrior",
    "P/L",
    # "likelihood",
    "profit_margin",
    "loss_margin",
]

DUMP_COL = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Vol",
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
    "OBV_UP",
    "sBuy",
    "sSell",
    "sProfit",
    "sPosition",
    "sCash",
    "sAvgCost",
    "xBuy",
    "xSell",
    "xProfit",
    "xPosition",
    "xCash",
    "xAvgCost",
    "xBuyOrder",
    "xSellOrder",
    "Kelly",
    "Postrior",
    "P/L",
    "likelihood",
    "profit_margin",
    "loss_margin",
]


def hunterPause(sp):
    if sp.simulate:
        return
    if sp.fetch_huobi:
        now = datetime.datetime.now()
        minutes_interval = INTERVAL_TO_MIN.get(sp.interval)
        next_whole_point = now.replace(second=0, microsecond=0) + datetime.timedelta(
            minutes=minutes_interval
        )
        if now.minute % minutes_interval != 0:
            extra_minutes = minutes_interval - now.minute % minutes_interval
            next_whole_point = now + datetime.timedelta(minutes=extra_minutes)
            next_whole_point = next_whole_point.replace(second=0, microsecond=0)
        sleep_seconds = (next_whole_point - now).total_seconds()
        seconds_to_wait = max(0, sleep_seconds) + 5  # Ensure non-negative value
        print(
            f"Will be start after: {seconds_to_wait} sec, {datetime.datetime.now()+datetime.timedelta(seconds=seconds_to_wait)}"
        )
        time.sleep(seconds_to_wait)


@dataclass
class HuntingStory:
    sensor: IMarketSensor
    scout: IStrategyScout
    engine: IEngine
    hunter: IHunter
    base_df: pd.DataFrame

    def pub_market_sensor(self, sp):
        def run_market_sensor():
            while True:
                k = self.sensor.fetch_one()
                print(k)
                dispatcher.send(signal="k_channel", message=k)
                hunterPause(sp)

        return threading.Thread(target=run_market_sensor)

    def move_forward(self, message):
        self.base_df = pd.concat(
            [self.base_df, message],
            ignore_index=True,
        )
        self.base_df = self.base_df[~self.base_df.index.duplicated(keep="last")]
        self.base_df = self.scout.market_recon(self.base_df)
        self.base_df = self.engine.hunt_plan(self.base_df)
        self.base_df = self.hunter.strike_phase(self.base_df)
        print(self.base_df[DEBUG_COL][-30:])
        return self.base_df, self.hunter.review_mission(self.base_df)

    def price_callback(self, message):
        self.base_df = pd.concat(
            [self.base_df, message],
            ignore_index=True,
        )
        self.base_df = self.base_df[~self.base_df.index.duplicated(keep="last")]
        self.base_df = self.scout.market_recon(self.base_df)
        self.base_df = self.engine.hunt_plan(self.base_df)
        self.base_df = self.hunter.strike_phase(self.base_df)
        print(self.base_df[DEBUG_COL][-30:])
        return self.base_df, self.hunter.review_mission(self.base_df)

def start_journey(sp):
    base_df = None
    sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)
    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)
    base_df = sensor.scan(2000 if not sp.simulate else 100)

    story = HuntingStory(sensor, scout, engine, hunter, base_df)
    dispatcher.connect(story.move_forward, signal="k_channel")

    pub_thread = story.pub_market_sensor(sp)
    print("start thread")
    pub_thread.start()
    pub_thread.join()


params = {
    "ATR_sample": 60,
    "atr_loss_margin": 3.5,
    "hard_cutoff": 0.95,
    "profit_loss_ratio": 3.0,
    "bayes_windows": 10,
    "lower_sample": 30.0,
    "upper_sample": 30.0,
    "interval": "1day",
    "funds": 100,
    "stake_cap": 10.5,
    "symbol": None,
    "surfing_level": 6,
    "fetch_huobi": True,
    "simulate": False,
}

if __name__ == "__main__":
    params.update(
        {
            "interval": "1min",
            "funds": 100,
            "stake_cap": 50,
            "symbol": Symbol("btcusdt"),
        }
    )
    sp = StrategyParam(**params)
    start_journey(sp)
