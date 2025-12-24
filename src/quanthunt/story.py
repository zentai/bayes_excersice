# Standard libraries
import datetime
import threading
import time
import traceback

# Third-party libraries
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict
from pydispatch import dispatcher
import copy

# Internal modules
from quanthunt.sensor.market_sensor import LocalMarketSensor, HuobiMarketSensor
from quanthunt.hunterverse.storage import HuntingCamp
from quanthunt.strategy.turtle_trading import (
    TurtleScout,
    buy_signal_from_mosaic_strategy,
)
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
    print(f"Pausing {sleep_seconds:.1f}s, until {next_time}")
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
    stop_flag: threading.Event = field(default_factory=threading.Event)

    def pub_market_sensor(self):
        while self.sensor.left():
            try:
                k = self.sensor.fetch_one()
                dispatcher.send(signal="k_channel", message=k)
                if self.stop_flag.is_set():
                    print("Stop fetching market data... (stop flag is set)")
                    break
                hunter_pause(self.params.interval)
            except Exception as e:
                print(f"Error in pub_market_sensor: {e}")
                print(traceback.format_exc())
                time.sleep(5)

    def move_forward(self, message):
        try:

            def current_buy_index(df: pd.DataFrame) -> int | None:
                """
                找出「最后一次卖出」之后仍未平仓的买入 index。
                若目前无持仓则回传 None。
                """
                # 1️⃣ 最近一次有卖出的行
                last_sell_idx = df.index[df["xSell"].notna()].max()

                # 2️⃣ 根据 last_sell_idx 切片；若从未卖出过，则从头搜
                if pd.isna(last_sell_idx):
                    sub_df = df
                else:
                    sub_df = df.loc[last_sell_idx:]

                # 3️⃣ 在剩余区段寻找 still-holding 的 buy
                holding_mask = (sub_df["xBuy"].notna()) & (sub_df["xSell"].isna())
                if holding_mask.any():
                    return sub_df[holding_mask].index[0]  # 取最早那笔买入
                return None

            def get_chandelier_exit_price(df: pd.DataFrame) -> float | None:
                """
                回传目前持仓的『动态止损线』（买入以来最高 exit_price）。
                若目前无持仓则回传 None。
                """
                # 第 1 步：找到当前持仓的 buy_index
                buy_idx = current_buy_index(df)
                if buy_idx is None:
                    return df.iloc[-1].exit_price  # 目前无持仓

                # 第 2 步：切片 [buy_idx : ]，取 exit_price 最大值
                stop_line = df.loc[buy_idx:, "exit_price"].max()
                return stop_line

            required_columns = set().union(*(h.columns for h in self.hunter.values()))
            message = pandas_util.equip_fields(message, list(required_columns))
            self.base_df = pd.concat([self.base_df, message], ignore_index=True)
            self.base_df = self.base_df[~self.base_df.index.duplicated(keep="last")]
            self.base_df = self.scout.market_recon(self.base_df)
            self.base_df = self.engine.hunt_plan(self.base_df)
            last_candlestick = self.base_df.iloc[-1]
            last_candlestick.exit_price = get_chandelier_exit_price(self.base_df)
            for hunter in self.hunter.values():
                hunter.load_memories(self.base_df)
                is_trend_gone = hunter.strike_phase(
                    lastest_candlestick=last_candlestick
                )
                if is_trend_gone:
                    from huobi.connection.subscribe_client import SubscribeClient

                    scheduler = SubscribeClient.subscribe_watch_dog.scheduler
                    if scheduler.running:
                        scheduler.remove_all_jobs()
                        scheduler.shutdown(wait=False)

                    print("⚠️ Terminating: trend ended + no position.")
                    self.stop_flag.set()
                    break  # 避免重复触发
            self._debug_actions(mission_status="Running")

        except Exception as e:
            print(f"Error in move_forward: {e}")
            print(traceback.format_exc())

    def _debug_actions(self, mission_status):
        if "statement" in self.params.debug_mode:
            print(self.base_df[self.debug_cols].tail(30))
        if "mission_review" in self.params.debug_mode:
            target_key = "s" if self.params.backtest else "x"
            # target_key = "s"
            if target_key in self.hunter:
                symbol = self.params.symbol.name.replace("usdt", "/usdt")
                client = target_key.upper()
                now = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"\n\033[32m✅ [{now}] {symbol} | {client} Review\033[0m")
                print(self.hunter[target_key].review_mission(self.base_df))
        if "statement_to_csv" in self.params.debug_mode:
            csv_path = f"{config.reports_dir}/{self.params}.csv"
            self.base_df[self.report_cols].to_csv(csv_path, index=False)
            print(f"CSV created: {csv_path}")

        if not self.params.backtest:
            pandas_util.write_status(
                sp=self.params,
                review_df=self.hunter["x"].review_mission(self.base_df),
                status=mission_status,
            )

    def callback_order_matched(
        self, client, order_id, order_status, price, position, execute_timestamp
    ):
        if client == "x":
            hunter = self.hunter[client]
            hunter.load_memories(self.base_df)
            self._debug_actions(mission_status="Running")
        hunter = self.hunter[client]
        self._debug_actions(mission_status="Running")

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
    scout = TurtleScout(params=sp, buy_signal_func=buy_signal_from_mosaic_strategy)
    engine = BayesianEngine(params=sp)

    bsp = copy.deepcopy(sp)
    bsp.funds = 1000000
    hunter = {
        "s": xHunter("s", params=bsp),
        "x": xHunter("x", params=sp, platform=Huobi("x", sp)),
        # "x": xHunter("x", params=sp),
    }
    base_df = scout.train(base_df)
    hunter["x"].load_memories(base_df)
    target_key = "s" if sp.backtest else "x"
    # target_key = "s"
    debug_cols = DEBUG_COL + hunter[target_key].columns
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

    story.pub_market_sensor()
    # pub_thread = story.pub_market_sensor()
    # pub_thread.start()
    # pub_thread.join()

    if "final_statement_to_csv" in sp.debug_mode:
        review = story.hunter["s"].review_mission(story.base_df)
        print(review)
        story.base_df[report_cols].to_csv(f"{config.reports_dir}/{sp}.csv", index=False)
        print(f"created: {config.reports_dir}/{sp}.csv")
    print("Mission accomplished. The hunt is over.")
    story._debug_actions(mission_status="Completed")
