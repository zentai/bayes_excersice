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


@dataclass
class HuntingStory:
    sensor: IMarketSensor
    scout: IStrategyScout
    engine: IEngine
    hunter: IHunter

    def move_forward(self, base_df):
        base_df = self.sensor.fetch(base_df)
        base_df = self.scout.market_recon(base_df)
        base_df = self.engine.hunt_plan(base_df)
        base_df = self.hunter.strike_phase(base_df)
        return base_df, self.hunter.review_mission(base_df)


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


def start_journey(sp):
    base_df = None

    def signal_handler(sig, frame):
        nonlocal base_df
        print("Received kill signal, retreating...")
        if base_df is not None:
            try:
                for order_type in (
                    "buy-stop-limit",
                    "buy-limit",
                    "buy-market",
                    "sell-limit",
                    "sell-stop-limit",
                    "sell-market",
                ):
                    success, fail = huobi_api.cancel_all_open_orders(
                        sp.symbol.name, order_type=OrderType.BUY_STOP_LIMIT
                    )
                    print(
                        f"cancel {order_type} orders success: {success}, fail: {fail}" 
                    )
                    if order_type in ("buy-stop-limit", "buy-limit", "buy-market"):
                        base_df.loc[
                            base_df.xBuyOrder.isin(success), "xBuyOrder"
                        ] = "Cancel"
                    elif order_type in ("sell-limit", "sell-stop-limit", "sell-market"):
                        base_df.loc[
                            base_df.xSellOrder.isin(success), "xSellOrder"
                        ] = "Cancel"
            except Exception as e:
                print(f"[Terminated] cancel process fail: {e}")

            base_df[DUMP_COL].to_csv(
                f"{config.reports_dir}/{sp.symbol.name}.csv", index=False
            )
            print(f"{config.reports_dir}/{sp.symbol.name}.csv")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if sp.simulate:
        sensor = LocalMarketSensor(symbol=sp.symbol, interval="local")
    elif sp.fetch_huobi:
        sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)

    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)
    # if not sp.simulate:
    #     hunter.load_memories()

    story = HuntingStory(sensor, scout, engine, hunter)
    base_df = sensor.scan(2000 if not sp.simulate else 100)
    final_review = None

    round = sensor.left() or 1000000
    for i in range(round):
        base_df, review = story.move_forward(base_df)
        final_review = review
        # print(base_df[DEBUG_COL][-30:])
        # print(f"{base_df.iloc[-1].Date}")
        # print(final_review)
        # base_df[DUMP_COL].to_csv(
        #     f"{config.reports_dir}/{sp.symbol.name}.csv", index=False
        # )
        # print(f"{config.reports_dir}/{sp.symbol.name}.csv")
        hunterPause(sp)
    base_df[DUMP_COL].to_csv(
        f"{config.reports_dir}/{sp.symbol.name}.csv", index=False
    )
    print(f"{config.reports_dir}/{sp.symbol.name}.csv")


        # try:
        #     base_df, review = story.move_forward(base_df)
        #     final_review = review
        #     print(base_df[DEBUG_COL][-30:])
        #     print(final_review)
        #     base_df[DUMP_COL].to_csv(
        #         f"{config.reports_dir}/{sp.symbol.name}.csv", index=False
        #     )
        #     print(f"{config.reports_dir}/{sp.symbol.name}.csv")
        #     hunterPause(sp)

        # except Exception as e:
        #     print(e)
        #     time.sleep(5)

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
@click.option("--ccy", default="bomeusdt", required=False, help="trade ccy pair")
@click.option(
    "--interval",
    required=False,
    default="1min",
    help="trade interval: 1min 5min 15min 30min 60min 4hour 1day 1mon 1week 1year",
)
@click.option(
    "--fund",
    required=False,
    type=int,
    default=100.0,
    help="initial funds",
)
@click.option(
    "--cap",
    required=False,
    type=int,
    default=50,
    help="Stake cap",
)
def main(ccy, interval, fund, cap):
    entry(ccy, interval, fund, cap)

def entry(ccy, interval, fund, cap):
    params.update({
        "interval": interval,
        "funds": fund,
        "stake_cap": cap,
        "symbol": Symbol(ccy),
    })
    sp = StrategyParam(**params)
    final_review = start_journey(sp)
    return final_review.iloc[-1].Profit


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
    "simulate": True,
}

if __name__ == "__main__":
    main()
