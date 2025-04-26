# Standard libraries
import datetime
import threading
import time

# Third-party libraries
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from pydispatch import dispatcher
import copy

# Internal modules
from quanthunt.sensor.market_sensor import LocalMarketSensor, HuobiMarketSensor
from quanthunt.hunterverse.storage import HuntingCamp
from quanthunt.strategy.turtle_trading import TurtleScout, emv_cross_strategy
from quanthunt.engine.probabilistic_engine import BayesianEngine
from quanthunt.utils import pandas_util
from quanthunt.hunterverse.interface import (
    IStrategyScout,
    IMarketSensor,
    IEngine,
    IHunter,
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


def hunter_pause(interval: str):
    now = datetime.datetime.now()
    minutes_interval = INTERVAL_TO_MIN.get(interval)
    next_interval = (now.minute // minutes_interval + 1) * minutes_interval
    next_time = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(
        minutes=next_interval
    )
    sleep_seconds = (next_time - now).total_seconds() + 5
    print(
        f"Next scan at {datetime.datetime.now() + datetime.timedelta(seconds=sleep_seconds)}\n"
    )
    time.sleep(sleep_seconds)


def to_hz(interval: str, value: float) -> float:
    minutes_interval = INTERVAL_TO_MIN.get(interval)
    return value / (minutes_interval * 60)


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
                    hunter_pause(sp.interval)
                except Exception as e:
                    print(f"error: {e}")

        return threading.Thread(target=run)

    def move_forward(self, message):
        try:
            # FIXMIE: we can use better logic
            for h in self.hunter.values():
                message = pandas_util.equip_fields(message, h.columns)
            message["Count_Hz"] = to_hz(self.params.interval, message.Count)
            message["Amount_Hz"] = to_hz(self.params.interval, message.Amount)
            self.base_df = pd.concat([self.base_df, message], ignore_index=True)
            self.base_df = self.base_df[~self.base_df.index.duplicated(keep="last")]
            self.base_df = self.scout.market_recon(self.base_df)
            self.base_df = self.engine.hunt_plan(self.base_df)
            for hunter in self.hunter.values():
                hunter.load_memories(self.base_df)
                hunter.strike_phase(lastest_candlestick=self.base_df.iloc[-1])
            self._debug_actions()

        except Exception as e:
            print(f"Error in move_forward: {e}")
            import traceback

            print(traceback.format_exc())

    def _debug_actions(self):
        if "statement" in self.params.debug_mode:
            print(self.base_df[self.debug_cols].tail(30))
        if "mission_review" in self.params.debug_mode:
            # FIXME: no need to loop all hunter
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

    def setup_dispatcher(self):
        dispatcher.connect(self.move_forward, signal="k_channel")
        for h in self.hunter.values():
            dispatcher.connect(
                self.callback_order_matched, signal=h.platform.TOPIC_ORDER_MATCHED
            )


def start_journey(sp):
    sensor = (
        LocalMarketSensor(symbol=sp.symbol, interval=sp.interval)
        if sp.backtest
        else HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)
    )
    camp = HuntingCamp(sp, sensor)
    base_df = camp.update()
    scout = TurtleScout(params=sp, buy_signal_func=emv_cross_strategy)
    engine = BayesianEngine(params=sp)

    bsp = copy.deepcopy(sp)
    bsp.funds = 1000000
    hunter = {
        "s": xHunter("s", params=sp),
        # "b": xHunter("b", params=sp),
        "x": xHunter("x", params=sp, platform=Huobi("x", sp)),
    }

    base_df = sensor.scan(2000 if not sp.backtest else 100)
    base_df["Count_Hz"] = to_hz(sp.interval, base_df.Count)
    base_df["Amount_Hz"] = to_hz(sp.interval, base_df.Amount)
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
    story.setup_dispatcher()

    pub_thread = story.pub_market_sensor(sp)
    pub_thread.start()
    pub_thread.join()

    if "final_statement_to_csv" in sp.debug_mode:
        review = story.hunter["s"].review_mission(story.base_df)
        print(review)
        story.base_df[report_cols].to_csv(f"{config.reports_dir}/{sp}.csv", index=False)
        print(f"created: {config.reports_dir}/{sp}.csv")

    return story.base_df[report_cols], story.hunter["s"].review_mission(story.base_df)
