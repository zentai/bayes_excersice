import sys
import os
import time
import click
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from utils import pandas_util
from settings import DATA_DIR, SRC_DIR, REPORTS_DIR

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
        print(self.hunter.review_mission(base_df))
        base_df[DUMP_COL].to_csv(
            f"{REPORTS_DIR}/{self.scout.params.symbol.name}_2.csv", index=False
        )
        print(f"{REPORTS_DIR}/{self.scout.params.symbol.name}_2.csv")

        seconds_to_wait = 60 - datetime.datetime.now().second + 5
        print(f"Will be start after: {seconds_to_wait} sec")
        time.sleep(seconds_to_wait)

        # print(f"Sleep: {self.sensor.interval_min} min")
        # time.sleep(self.sensor.interval_min * 60)

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
class Symbol:
    name: str
    amount_prec: float = 4
    price_prec: float = 4

    def __post_init__(self):
        from huobi.client.generic import GenericClient

        client = GenericClient()
        for item in client.get_exchange_symbols():
            if item.symbol == self.name:
                self.amount_prec = item.amount_precision
                self.price_prec = item.price_precision

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name


@dataclass
class StrategyParam:
    symbol: str
    ATR_sample: int = 7
    atr_loss_margin: float = 1.5
    bayes_windows: int = 30
    lower_sample: int = 7
    upper_sample: int = 7
    interval: str = "1min"
    fetch_huobi: bool = False
    simulate: bool = False

    def __post_init__(self):
        self.ATR_sample = int(self.ATR_sample)
        self.bayes_windows = int(self.bayes_windows)
        self.lower_sample = int(self.lower_sample)
        self.upper_sample = int(self.upper_sample)


@click.command()
@click.option("--ccy", required=True, help="trade ccy pair")
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
    import sys
    import os

    from settings import DATA_DIR, SRC_DIR, REPORTS_DIR
    from strategy.turtle_trading import TurtleScout
    from engine.probabilistic_engine import BayesianEngine
    from sensor.market_sensor import LocalMarketSensor
    from sensor.market_sensor import HuobiMarketSensor
    from tradingfirm.trader import xHunter

    params = {
        "ATR_sample": 30,
        "atr_loss_margin": 1.5,
        "bayes_windows": 30,
        "lower_sample": 30,
        "upper_sample": 30,
        "interval": "1min",
        "symbol": Symbol(ccy),
        "fetch_huobi": True,
        "simulate": True,
    }
    sp = StrategyParam(**params)
    # sensor = LocalMarketSensor(symbol=sp.symbol, interval='local')
    sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)

    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)
    # gains_bag = GainsBags(init_fund=100, position=0)

    # Adjust start time
    seconds_to_wait = 60 - datetime.datetime.now().second + 5
    print(f"Will be start after: {seconds_to_wait} sec")
    time.sleep(seconds_to_wait)

    story = HuntingStory(sensor, scout, engine, hunter)
    base_df = sensor.scan(2000)
    for i in range(20000):
        try:
            base_df = story.start(base_df)
        except Exception as e:
            print(e)
            time.sleep(3)


if __name__ == "__main__":
    main()
