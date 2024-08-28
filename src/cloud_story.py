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

from .utils import pandas_util
from .hunterverse.interface import IStrategyScout
from .hunterverse.interface import IMarketSensor
from .hunterverse.interface import IEngine
from .hunterverse.interface import IHunter
from .hunterverse.interface import Symbol
from .hunterverse.interface import StrategyParam
from .hunterverse.interface import INTERVAL_TO_MIN

from .sensor.market_sensor import LocalMarketSensor
from .sensor.market_sensor import HuobiMarketSensor
from .tradingfirm.pubsub_trader import xHunter
from .tradingfirm.pubsub_trader import (
    HUNTER_COLUMNS,
    BUY_FILLED,
    SELL_FILLED,
    CUTOFF_FILLED,
)

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir

# from .tradingfirm.trader import xHunter
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
    # "time_cost",
    "buy",
    "sell",
    "profit",
    "OBV_UP",
    "sBuy",
    "sSell",
    "sProfit",
    "sPosition",
    "sCash",
    "sAvgCost",
    "sPnLRatio",
    # "xBuy",
    # "xSell",
    # "xProfit",
    # "xPosition",
    # "xCash",
    # "xAvgCost",
    # "xBuyOrder",
    # "xSellOrder",
    "Kelly",
    # "Postrior",
    "P/L",
    "likelihood",
    # "profit_margin",
    # "loss_margin",
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
    "sPnLRatio",
    # "xBuy",
    # "xSell",
    # "xProfit",
    # "xPosition",
    # "xCash",
    # "xAvgCost",
    # "xBuyOrder",
    # "xSellOrder",
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
                dispatcher.send(signal="k_channel", message=k)
                hunterPause(sp)

        return threading.Thread(target=run_market_sensor)

    def move_forward(self, message):
        message = pandas_util.equip_fields(message, HUNTER_COLUMNS)
        self.base_df = pd.concat(
            [self.base_df, message],
            ignore_index=True,
        )
        self.base_df = self.base_df[~self.base_df.index.duplicated(keep="last")]
        self.base_df = self.scout.market_recon(self.base_df)
        self.base_df = self.engine.hunt_plan(self.base_df)

        def _build_hunting_cmd(lastest_candlestick):
            print(f"kelly: {lastest_candlestick.Kelly}")
            _High = lastest_candlestick.High
            _Low = lastest_candlestick.Low
            hunting_command = {}
            hunting_command["sell"] = {
                "hunting_id": lastest_candlestick.Date,
                "exit_price": lastest_candlestick.exit_price,
                "Stop_profit": lastest_candlestick.Stop_profit,
                "market_High": _High,
                "market_Low": _Low,
            }
            buy_signal = lastest_candlestick.OBV_UP == True
            # buy_signal = lastest_candlestick.BuySignal == 1
            if buy_signal:

                price = self.base_df.tail(self.hunter.params.upper_sample).High.max()
                order_type = "B" if price == lastest_candlestick.High else "BL"

                hunting_command["buy"] = {
                    "hunting_id": lastest_candlestick.Date,
                    "target_price": price,
                    "order_type": order_type,
                    "kelly": 1,  # lastest_candlestick.Kelly,
                    "market_High": _High,
                    "market_Low": _Low,
                }
            return hunting_command

        hunting_command = _build_hunting_cmd(lastest_candlestick=self.base_df.iloc[-1])
        self.hunter.strike_phase(hunting_command)
        # print(self.base_df[DEBUG_COL][-30:])
        print(self.hunter.review_mission(self.base_df))
        # return self.base_df, self.hunter.review_mission(self.base_df)

    def sim_attack_feedback(self, order_id, order_status, price, position):
        if order_status in (BUY_FILLED):
            s_buy_order = self.base_df.Date == order_id
            self.base_df.loc[s_buy_order, "sBuyOrder"] = order_id
            self.base_df.loc[s_buy_order, "sBuy"] = price
            self.base_df.loc[s_buy_order, "sPosition"] = self.hunter.sim_bag.position
            self.base_df.loc[s_buy_order, "sCash"] = self.hunter.sim_bag.cash
            self.base_df.loc[s_buy_order, "sAvgCost"] = self.hunter.sim_bag.avg_cost
        elif order_status in (SELL_FILLED, CUTOFF_FILLED):
            s_sell_order = self.base_df.Date == order_id
            self.base_df.loc[s_sell_order, "sSellOrder"] = order_id
            self.base_df.loc[s_sell_order, "sBuy"] = self.hunter.sim_bag.avg_cost
            self.base_df.loc[s_sell_order, "sSell"] = price
            profit = (price / self.hunter.sim_bag.avg_cost) - 1
            self.base_df.loc[s_sell_order, "sProfit"] = profit
            self.base_df.loc[s_sell_order, "sPosition"] = self.hunter.sim_bag.position
            self.base_df.loc[s_sell_order, "sCash"] = self.hunter.sim_bag.cash
            self.base_df.loc[s_sell_order, "sAvgCost"] = self.hunter.sim_bag.avg_cost
            self.base_df.loc[s_sell_order, "sPnLRatio"] = profit / (
                1 - self.hunter.params.hard_cutoff
            )


def start_journey(sp):
    base_df = None
    # sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)
    sensor = LocalMarketSensor(symbol=sp.symbol, interval=sp.interval)
    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)
    base_df = sensor.scan(2000 if not sp.simulate else 100)
    story = HuntingStory(sensor, scout, engine, hunter, base_df)
    dispatcher.connect(story.move_forward, signal="k_channel")
    dispatcher.connect(story.sim_attack_feedback, signal="sim_attack_feedback")
    pub_thread = story.pub_market_sensor(sp)
    print("start thread")
    pub_thread.start()
    pub_thread.join()
    print(story.base_df)
    story.base_df[DUMP_COL].to_csv(f"{REPORTS_DIR}/{sp.symbol.name}.csv", index=False)
    print(f"created: {REPORTS_DIR}/{sp.symbol.name}.csv")


params = {
    "ATR_sample": 7,
    "atr_loss_margin": 1.5,
    "hard_cutoff": 0.95,
    "profit_loss_ratio": 3.0,
    "bayes_windows": 30,
    "lower_sample": 30.0,
    "upper_sample": 30.0,
    "interval": "1min",
    "funds": 100,
    "stake_cap": 50,
    "symbol": None,
    "surfing_level": 7,
    "fetch_huobi": False,
    "simulate": True,
}

if __name__ == "__main__":
    params.update(
        {
            "interval": "1day",
            "funds": 100,
            "stake_cap": 50,
            "symbol": Symbol("nvda"),
        }
    )
    sp = StrategyParam(**params)
    print(sp)
    start_journey(sp)
