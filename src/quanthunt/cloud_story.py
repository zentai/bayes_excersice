# Standard libraries
import datetime
import threading
import time
import signal
import sys
import copy

# Third-party libraries
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from pydispatch import dispatcher
import click

# Internal modules
from quanthunt.strategy.turtle_trading import TurtleScout, emv_cross_strategy
from quanthunt.engine.probabilistic_engine import BayesianEngine
from quanthunt.utils import pandas_util
from quanthunt.hunterverse.interface import (
    IStrategyScout,
    IMarketSensor,
    IEngine,
    IHunter,
    Symbol,
    StrategyParam,
    INTERVAL_TO_MIN,
    DEBUG_COL,
    DUMP_COL,
)
from quanthunt.sensor.market_sensor import (
    LocalMarketSensor,
    HuobiMarketSensor,
)
from quanthunt.tradingfirm.xtrader import (
    xHunter,
    Huobi,
)
from quanthunt.config.core_config import config


def hunter_pause(sp):
    if sp.backtest:
        return
    now = datetime.datetime.now()
    minutes_interval = INTERVAL_TO_MIN.get(sp.interval)
    next_interval = (now.minute // minutes_interval + 1) * minutes_interval
    next_time = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(
        minutes=next_interval
    )
    sleep_seconds = (next_time - now).total_seconds() + 5  # add extra 5 seconds buffer
    print(
        f"Next scan at {datetime.datetime.now() + datetime.timedelta(seconds=sleep_seconds)}\n"
    )
    time.sleep(sleep_seconds)


@dataclass
class HuntingStory:
    params: StrategyParam
    sensor: IMarketSensor
    scout: IStrategyScout
    engine: IEngine
    hunter: Dict[str, IHunter]
    base_df: pd.DataFrame
    debug_cols: str
    report_cols: str

    def pub_market_sensor(self, sp):
        def run():
            while self.sensor.left():
                try:
                    k = self.sensor.fetch_one()
                    dispatcher.send(signal="k_channel", message=k)
                    hunter_pause(sp)
                except Exception as e:
                    print(f"error: {e}")

        return threading.Thread(target=run)

    def move_forward(self, message):
        """Process incoming market data and update internal state."""
        try:
            # Equip fields across all hunters
            for h in self.hunter.values():
                message = pandas_util.equip_fields(message, h.columns)

            # Concatenate and remove duplicated rows
            self.base_df = pd.concat([self.base_df, message], ignore_index=True)
            self.base_df = self.base_df[~self.base_df.index.duplicated(keep="last")]

            # Update market information with scout and engine processing
            self.base_df = self.scout.market_recon(self.base_df)
            self.base_df = self.engine.hunt_plan(self.base_df)

            # Process each hunter's phases
            for hunter in self.hunter.values():
                hunter.load_memories(self.base_df)
                hunter.strike_phase(lastest_candlestick=self.base_df.iloc[-1])

            # Debug logging or CSV export based on debug_mode settings
            self._debug_actions()

        except Exception as e:
            print(f"Error in move_forward: {e}")
            import traceback

            print(traceback.format_exc())

    def _debug_actions(self):
        """Handles debug outputs and CSV export if set in debug_mode."""
        if "statement" in self.params.debug_mode:
            print(self.base_df[self.debug_cols].tail(30))
        if "mission_review" in self.params.debug_mode:
            # 假設每個hunter皆有review_mission方法
            for hunter in self.hunter.values():
                print(hunter.review_mission(self.base_df))
        if "statement_to_csv" in self.params.debug_mode:
            csv_path = f"{config.reports_dir}/{self.params}.csv"
            self.base_df[self.report_cols].to_csv(csv_path, index=False)
            print(f"CSV created: {csv_path}")

    def callback_order_matched(
        self, client, order_id, order_status, price, position, execute_timestamp
    ):
        if client == "x":
            print(f"received {order_id} update")
            hunter = self.hunter[client]
            hunter.load_memories(self.base_df)
            self._debug_actions()


def start_journey(sp):
    base_df = None
    # camp = HuntingCamp(sp)
    # load_df = camp.load()
    load_df = pd.DataFrame()
    if sp.backtest:
        sensor = LocalMarketSensor(symbol=sp.symbol, interval=sp.interval)
    else:
        sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)

    # 移除 load_df 中已存在的日期部分
    update_df = sensor.scan(2000 if not sp.backtest else 100)
    if not load_df.empty:
        update_df = update_df[~update_df["Date"].isin(load_df["Date"])]

    # 合并 load_df 和 update_df，成为 base_df
    base_df = pd.concat([load_df, update_df], ignore_index=True)

    scout = TurtleScout(params=sp, buy_signal_func=emv_cross_strategy)
    engine = BayesianEngine(params=sp)
    import copy

    bsp = copy.deepcopy(sp)
    bsp.funds = 1000000
    hunter = {
        "s": xHunter("s", params=sp),
        # "b": xHunter("b", params=sp),
        "x": xHunter("x", params=sp, platform=Huobi("x", sp)),
    }

    base_df = sensor.scan(2000 if not sp.backtest else 100)
    base_df = scout.train(base_df)
    hunter["x"].load_memories(base_df)

    debug_cols = DEBUG_COL + sum(
        [h.columns for h in hunter.values() if h.client == "x"], []
    )
    report_cols = DUMP_COL + sum([h.columns for h in hunter.values()], [])

    story = HuntingStory(
        sp,
        sensor,
        scout,
        engine,
        hunter,
        base_df,
        debug_cols=debug_cols,
        report_cols=report_cols,
    )
    dispatcher.connect(story.move_forward, signal="k_channel")
    for h in hunter.values():
        dispatcher.connect(
            story.callback_order_matched, signal=h.platform.TOPIC_ORDER_MATCHED
        )
    pub_thread = story.pub_market_sensor(sp)
    pub_thread.start()
    pub_thread.join()
    # print(story.base_df[report_cols])
    # review = story.hunter.review_mission(story.base_df)
    # sensor.db.save(collection_name=f"{sp.symbol.name}_review", df=review)
    if "final_statement_to_csv" in sp.debug_mode:
        review = story.hunter["s"].review_mission(story.base_df)
        print(review)
        story.base_df[report_cols].to_csv(f"{config.reports_dir}/{sp}.csv", index=False)
        print(f"created: {config.reports_dir}/{sp}.csv")
    # visualize_backtest(story.base_df)
    return story.base_df[report_cols], story.hunter["s"].review_mission(story.base_df)
