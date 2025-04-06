import signal
import sys
import threading
import time
import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Dict

from config import config
from huobi.constant.definition import OrderType

from .strategy.turtle_trading import TurtleScout, emv_cross_strategy
from .engine.probabilistic_engine import BayesianEngine

from .utils import pandas_util
from .hunterverse.interface import IStrategyScout
from .hunterverse.interface import IMarketSensor
from .hunterverse.interface import IEngine
from .hunterverse.interface import IHunter
from .hunterverse.interface import Symbol
from .hunterverse.interface import StrategyParam
from .hunterverse.interface import INTERVAL_TO_MIN
from .hunterverse.interface import xBuyOrder, xSellOrder

# from .hunterverse.storage import HuntingCamp

from .sensor.market_sensor import LocalMarketSensor
from .sensor.market_sensor import HuobiMarketSensor
from .sensor.market_sensor import MongoMarketSensor

# from .tradingfirm.pubsub_trader import xHunter
from .tradingfirm.xtrader import xHunter, Huobi
from .tradingfirm.xtrader import (
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
    # "Low",
    "Close",
    # "turtle_h",
    # "ema_short",
    # "ema_long",
    "BuySignal",
    "Stop_profit",
    "exit_price",
    # "time_cost",
    # "buy",
    # "sell",
    # "profit",
    # "OBV",
    # "OBV_UP",
    # "Matured",
    # "Kelly",
    # "Postrior",
    # "P/L",
    # "likelihood",
    # "profit_margin",
    # "loss_margin",
    # "+DI",
    # "-DI",
    # "ADX_Signed",
    # "drift",
    # "volatility",
    # "pred_price",
    "Slope",
    "Kalman",
    "HMM_State",
    "UP_State",
]

DUMP_COL = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "HMM_State",
    "Kalman",
    "Vol",
    "turtle_h",
    "BuySignal",
    "ema_short",
    "ema_long",
    "Stop_profit",
    "exit_price",
    "Matured",
    "time_cost",
    "buy",
    "sell",
    "profit",
    "P/L",
    # "turtle_l",
    # "turtle_h",
    # "OBV",
    # "OBV_UP",
    # "upper_bound",
    # "lower_bound",
    # "Kelly",
    # "Postrior",
    # "likelihood",
    # "profit_margin",
    # "loss_margin",
    # "+DM",
    # "-DM",
    # "+DI",
    # "-DI",
    # "ADX_Signed",
    "drift",
    "volatility",
    "pred_price",
    "log_returns",
    "volatility",
    "global_log_volatility",
    "global_log_vol",
    "KReturnVol",
    "RVolume",
    "UP_State",
    "Slope",
]


def hunterPause(sp):
    if sp.backtest:
        return
    else:
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
        print()
        time.sleep(seconds_to_wait)


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
        def run_market_sensor():
            while self.sensor.left():
                try:
                    k = self.sensor.fetch_one()
                    dispatcher.send(signal="k_channel", message=k)
                    hunterPause(sp)
                except Exception as e:
                    print(f"error: {e}")

        return threading.Thread(target=run_market_sensor)

    def move_forward(self, message):
        try:
            for h in self.hunter.values():
                message = pandas_util.equip_fields(message, h.columns)
            self.base_df = pd.concat(
                [self.base_df, message],
                ignore_index=True,
            )
            self.base_df = self.base_df[~self.base_df.index.duplicated(keep="last")]
            self.base_df = self.scout.market_recon(self.base_df)
            self.base_df = self.engine.hunt_plan(self.base_df)

            for _, hunter in self.hunter.items():
                hunter.load_memories(self.base_df, deals=self.params.load_deals)
                hunter.strike_phase(lastest_candlestick=self.base_df.iloc[-1])

            if "statement" in self.params.debug_mode:
                print(self.base_df[self.debug_cols][-30:])
            if "mission_review" in self.params.debug_mode:
                print(hunter.review_mission(self.base_df))
            if "statement_to_csv" in self.params.debug_mode:
                self.base_df[self.report_cols].to_csv(
                    f"{REPORTS_DIR}/{self.params}.csv", index=False
                )
                print(f"created: {REPORTS_DIR}/{self.params}.csv")
        except Exception as e:
            print(e)
            import traceback

            print(traceback.format_exc())

    def callback_order_matched(
        self, client, order_id, order_status, price, position, execute_timestamp
    ):
        if client == "x":
            print(f"received {order_id} update")
            hunter = self.hunter[client]
            hunter.load_memories(self.base_df, deals=self.params.load_deals)
            if "statement" in self.params.debug_mode:
                print(self.base_df[self.debug_cols][-30:])
            if "mission_review" in self.params.debug_mode:
                print(hunter.review_mission(self.base_df))
            if "statement_to_csv" in self.params.debug_mode:
                self.base_df[self.report_cols].to_csv(
                    f"{REPORTS_DIR}/{self.params}.csv", index=False
                )
                print(f"created: {REPORTS_DIR}/{self.params}.csv")


def start_journey(sp):
    base_df = None
    # camp = HuntingCamp(sp)
    # load_df = camp.load()
    load_df = pd.DataFrame()
    if sp.backtest:
        sensor = LocalMarketSensor(symbol=sp.symbol, interval=sp.interval)
        # sensor = MongoMarketSensor(symbol=sp.symbol, interval=sp.interval)
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
    hunter["x"].load_memories(base_df, deals=sp.load_deals)

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
        story.base_df[report_cols].to_csv(f"{REPORTS_DIR}/{sp}.csv", index=False)
        print(f"created: {REPORTS_DIR}/{sp}.csv")
    # visualize_backtest(story.base_df)
    return story.base_df[report_cols], story.hunter["s"].review_mission(story.base_df)


import matplotlib.pyplot as plt
import seaborn as sns


def visualize_backtest(df, window_size=60):
    # Ensure Date and Matured columns are in datetime format
    df["Date"] = pd.to_datetime(df["Date"])
    df["Matured"] = pd.to_datetime(df["Matured"])

    # Set up the visualization style
    sns.set(style="whitegrid")

    # Capture Rate Calculation
    high_profit_trades = (df["P/L"] > 3).sum()
    captured_high_profit_trades = ((df["BuySignal"] == True) & (df["P/L"] > 3)).sum()
    capture_rate = (
        captured_high_profit_trades / high_profit_trades
        if high_profit_trades > 0
        else 0
    )

    # Signal Coverage Rate Calculation
    total_trading_days = len(df)
    signal_days = df["BuySignal"].fillna(False).astype(int).sum()
    signal_coverage_rate = (
        signal_days / total_trading_days if total_trading_days > 0 else 0
    )

    # Average Holding Period Calculation
    df["holding_period"] = (
        df["Matured"] - df["Date"]
    ).dt.total_seconds() / 3600  # Convert to hours
    average_holding_period = (
        df["holding_period"].mean() if not df["holding_period"].isna().all() else 0
    )

    # Max Consecutive Wins/Losses Calculation using pandas
    df["win"] = df["P/L"] > 0
    df["loss"] = df["P/L"] <= 0
    # Calculate consecutive wins and losses
    df["consecutive_wins"] = (
        df["win"]
        .astype(int)
        .groupby((df["win"] != df["win"].shift()).cumsum())
        .cumsum()
    )
    df["consecutive_losses"] = (
        df["loss"]
        .astype(int)
        .groupby((df["loss"] != df["loss"].shift()).cumsum())
        .cumsum()
    )
    # Get maximum consecutive wins and losses
    max_consecutive_wins = df["consecutive_wins"].max()
    max_consecutive_losses = df["consecutive_losses"].max()

    # Win Rate Trend Calculation using moving window
    df["win_numeric"] = df["win"].astype(int)
    win_rate_trend = df["win_numeric"].rolling(window=window_size, min_periods=1).mean()

    # Expected Value and Breakeven Point Calculation
    win_rate = df["win"].mean() if not df["win"].isna().all() else 0
    fail_rate = 1 - win_rate

    avg_profit = df.loc[df["P/L"] > 0, "P/L"].mean() if (df["P/L"] > 0).any() else 0
    avg_loss = -df.loc[df["P/L"] <= 0, "P/L"].mean() if (df["P/L"] <= 0).any() else 0

    expected_value = (win_rate * avg_profit) - (fail_rate * avg_loss)

    profit_loss_ratio = avg_profit / avg_loss if avg_loss != 0 else 0
    breakeven_win_rate = 1 / (1 + profit_loss_ratio) if profit_loss_ratio != 0 else 0

    # Print summary metrics
    print("Summary Metrics:")
    print(f"Capture Rate: {capture_rate:.3f}")
    print(f"Signal Coverage Rate: {signal_coverage_rate:.3f}")
    print(f"Average Holding Period (hours): {average_holding_period:.2f}")
    print(f"Max Consecutive Wins: {max_consecutive_wins}")
    print(f"Max Consecutive Losses: {max_consecutive_losses}")
    print(f"Expected Value per Trade: {expected_value:.3f}")
    print(f"Breakeven Win Rate: {breakeven_win_rate:.3f}")

    # P/L Binning into 10 segments
    pl_min = 0
    pl_max = df["P/L"].max()
    bins = np.linspace(
        pl_min, pl_max, 11
    )  # Split into 10 parts (11 points create 10 bins)
    print(bins)
    df["P/L_bins"] = pd.cut(df["P/L"], bins=bins, include_lowest=True)

    # Count occurrences for each bin using P/L_bins
    captured_counts = df.loc[df["BuySignal"] == True].groupby("P/L_bins").size()
    total_counts = df.groupby("P/L_bins").size()

    # Combine into a summary DataFrame
    summary_df = pd.DataFrame(
        {
            "P/L_bin": captured_counts.index.astype(str),
            "Captured_Counts": captured_counts,
            "Total_Counts": total_counts,
        }
    ).fillna(0)

    # Print the summary DataFrame
    print("Summary of Captured vs Total Counts per P/L Bin:")
    print(summary_df)

    # Profit and Loss Distribution Visualization (side-by-side bar chart)
    x = np.arange(len(captured_counts.index))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2, captured_counts, width, label="Captured by BuySignal", alpha=0.6
    )
    bars2 = ax.bar(
        x + width / 2, total_counts, width, label="Total High P/L", alpha=0.3
    )

    # Add some text for labels, title and axes ticks
    ax.set_xlabel("P/L Bins")
    ax.set_ylabel("Number of Trades")
    ax.set_title("P/L Distribution for Captured vs. Total High Profit Trades")
    ax.set_xticks(x)
    ax.set_xticklabels(captured_counts.index.astype(str), rotation=45)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

    # Calculating Capture Rate by bin
    capture_rate_by_bin = (captured_counts / total_counts).fillna(0)

    # Capture Rate Trend Over P/L Bins Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(
        capture_rate_by_bin.index.astype(str),
        capture_rate_by_bin,
        marker="o",
        linestyle="-",
        color="g",
    )
    plt.title("Capture Rate Trend Over P/L Bins")
    plt.xlabel("P/L Bins")
    plt.ylabel("Capture Rate")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Win Rate Trend Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], win_rate_trend, marker="o", linestyle="-", color="b")
    plt.title("Win Rate Trend Over Time (Moving Window)")
    plt.xlabel("Date")
    plt.ylabel("Win Rate")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# very good for BTCUSDT day K
# params = {
#     "ATR_sample": 15,
#     "atr_loss_margin": 3,
#     "hard_cutoff": 0.9,
#     "profit_loss_ratio": 2,
#     "bayes_windows": 15,
#     "lower_sample": 15.0,
#     "upper_sample": 15.0,
#     "interval": "1min",
#     "funds": 100,
#     "stake_cap": 50,
#     "symbol": None,
#     "surfing_level": 3,
#     "fetch_huobi": False,
#     "simulate": True,
# }


params = {
    # Buy
    "ATR_sample": 60,
    "bayes_windows": 10,
    "lower_sample": 60,
    "upper_sample": 60,
    # Sell
    "hard_cutoff": 0.9,
    "profit_loss_ratio": 3,
    "atr_loss_margin": 3,
    "surfing_level": 7,
    # Period
    "interval": "1day",
    "funds": 100,
    "stake_cap": 100,
    "symbol": None,
    "backtest": True,
    "debug_mode": [
        # "statement",
        # "statement_to_csv",
        # "mission_review",
        "final_statement_to_csv",
    ],
}
import click
from typing import List


@click.command()
@click.option("--symbol", default="moveusdt", help="Trading symbol (e.g. trxusdt)")
@click.option("--interval", default="1min", help="Trading interval")
@click.option("--funds", default=50.2, type=float, help="Available funds")
@click.option("--cap", default=10.1, type=float, help="Stake cap")
@click.option(
    "--deals",
    default="",
    help="Comma separated deal IDs",
)
@click.option(
    "--start_deal",
    default=0,
    type=int,
    help="start to load from deal id",
)
def main(
    symbol: str, interval: str, funds: float, cap: float, deals: str, start_deal: int
):
    deal_ids = [int(x.strip()) for x in deals.split(",") if x.strip()] if deals else []

    params.update(
        {
            "funds": funds,
            "stake_cap": cap,
            "symbol": Symbol(symbol),
            "interval": interval,
            "backtest": False,
            "debug_mode": [
                "statement",
                "statement_to_csv",
                "mission_review",
                "final_statement_to_csv",
            ],
            "load_deals": deal_ids,
            "start_deal": start_deal,
            "api_key": "fefd13a1-bg2hyw2dfg-440b3c64-576f2",
            "secret_key": "1a437824-042aa429-0beff3ba-03e26",
        }
    )
    sp = StrategyParam(**params)
    start_journey(sp)


if __name__ == "__main__":
    main()
