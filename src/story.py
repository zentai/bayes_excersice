import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from utils import pandas_util

DEBUG_COL = [
    "Date",
    # "Open",
    # "High",
    # "Low",
    "Close",
    # "BuySignal",
    # "Stop_profit",
    "exit_price",
    "Matured",
    "time_cost",
    "buy",
    "sell",
    "profit",
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
    "Kelly",
    "Postrior",
    "P/L",
    "likelihood",
    "profit_margin",
    "loss_margin",
]


class IMarketSensor(ABC):
    def __init__(self, symbol, interval):
        self.symbol = symbol
        self.interval = interval
        self.interval_min = pandas_util.INTERVAL_TO_MIN.get(interval)

    @abstractmethod
    def scan(self, limits):
        pass

    @abstractmethod
    def fetch(self, base_df):
        pass


class IStrategyScout(ABC):
    @abstractmethod
    def market_recon(self, mission_blueprint):
        pass


class IEngine(ABC):
    @abstractmethod
    def hunt_plan(self, base_df):
        pass


class IHunter(ABC):
    @abstractmethod
    def strike_phase(self, base_df):
        pass


@dataclass
class HuntingStory:
    sensor: IMarketSensor
    scout: IStrategyScout
    engine: IEngine
    hunter: IHunter
    # gains_bag: "GainsBag"
    # mission_blueprint: "MissionBlueprint"
    # capital_trap: "CapitalTrap"

    def start(self, base_df):
        base_df = self.scout.market_recon(base_df)
        base_df = self.engine.hunt_plan(base_df)
        base_df = self.hunter.strike_phase(base_df)

        print(base_df[DEBUG_COL][-50:])
        base_df[DUMP_COL].to_csv("hunt.csv")
        print("hunt.csv")
        print(f"Sleep: {self.sensor.interval_min} min")
        time.sleep(self.sensor.interval_min * 60)
        base_df = self.sensor.fetch(base_df)
        return base_df
        # hunt_plan.to_csv("hunt.csv")
        # print("hunt.csv")
        # self.hunter.execute_trade(hunt_plan, self.capital_trap)
        # trade_result = self.capital_trap.monitor_trade()
        # self.gains_bag.update(trade_result)
        # self.hunter.observe_trade(self.capital_trap)
        # self.hunter.review_mission()


@dataclass
class StrategyParam:
    ATR_sample: int = 7
    atr_loss_margin: float = 1.5
    bayes_windows: int = 30
    lower_sample: int = 7
    upper_sample: int = 7
    interval: str = "1min"

    def __post_init__(self):
        self.ATR_sample = int(self.ATR_sample)
        self.bayes_windows = int(self.bayes_windows)
        self.lower_sample = int(self.lower_sample)
        self.upper_sample = int(self.upper_sample)


if __name__ == "__main__":
    import sys
    import os

    from settings import DATA_DIR, SRC_DIR, REPORTS_DIR
    from strategy.turtle_trading import TurtleScout
    from engine.probabilistic_engine import BayesianEngine
    from sensor.market_sensor import LocalMarketSensor
    from sensor.market_sensor import HuobiMarketSensor
    from tradingfirm.trader import xHunter

    params = {
        "ATR_sample": 60,
        "atr_loss_margin": 1.5,
        "bayes_windows": 30,
        "lower_sample": 30,
        "upper_sample": 30,
        "interval": "1min",
    }
    sp = StrategyParam(**params)
    # sensor = LocalMarketSensor(symbol="btcusdt", interval="1min")
    sensor = HuobiMarketSensor(symbol="btcusdt", interval=sp.interval)

    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)
    # gains_bag = GainsBags(init_fund=100, position=0)

    story = HuntingStory(sensor, scout, engine, hunter)
    base_df = sensor.scan(2000)
    for i in range(2000):
        base_df = story.start(base_df)
