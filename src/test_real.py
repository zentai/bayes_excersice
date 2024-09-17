import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import signal
import sys
import time
import click
import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np
from .utils import pandas_util
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
from .story import HuntingStory, hunterPause

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
    "symbol": Symbol("flokiusdt"),
    "surfing_level": 6,
    "fetch_huobi": True,
    "simulate": True,
}

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

    print(sp)
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
        hunterPause(sp)
    base_df[DUMP_COL].to_csv(
        f"{config.reports_dir}/{sp.symbol.name}.csv", index=False
    )
    print(f"{config.reports_dir}/{sp.symbol.name}.csv")
    return final_review

def chart(df, code, axlist):
    # Ensure the index is of type DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
    
    # Create columns for scatter plots with NaN where there's no marker
    obv_up_marker = np.where(df["OBV_UP"], df["OBV"], np.nan)
    close_up_marker = np.where(df["OBV_UP"], df["Close"], np.nan)

    if 'sCash' in df.columns:
        df['sCash'].fillna(method='ffill', inplace=True)

    # Prepare additional plots
    fb = dict(y1=df['lower_bound'].values, y2=df['upper_bound'].values)
    [ax.clear() for ax in axlist]
    price_area = axlist[0]
    obv_area = axlist[1]
    fund_area = axlist[2]
    add_plots = [
        mpf.make_addplot(close_up_marker, ax=price_area, type='scatter', markersize=100, marker='^', color='y'),
        mpf.make_addplot(df["OBV"], ax=obv_area, panel=1, fill_between=fb, color='r', secondary_y=False, ylabel='OBV', alpha=0.1),
        mpf.make_addplot(obv_up_marker, ax=obv_area, panel=1, type='scatter', markersize=100, marker='^', color='red'),
        mpf.make_addplot(df["sCash"], ax=fund_area, panel=2, color='b', secondary_y=False, ylabel='sCash'),
    ]
    mpf.plot(df, type='candle', ax=price_area, addplot=add_plots, title=f"{code} Price and OBV with Bounds", ylabel='Price (USD)', style='yahoo', datetime_format='%Y-%m-%d %H:%M:%S')


def main(sp):
    # 初始化图表
    plt.ion()  # 开启交互模式

    sensor = LocalMarketSensor(symbol=sp.symbol, interval="local")
    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)
    # if not sp.simulate:
    #     hunter.load_memories()
    story = HuntingStory(sensor, scout, engine, hunter)
    df = sensor.scan(100)
    # fig, axlist = mpf.plot(df, type='candle', returnfig=True, style='yahoo')
    # ax = axlist[0]

    fig, axlist = plt.subplots(3, 1, sharex=True, figsize=(10, 8))  # 创建三个共享x轴的子图
    round = sensor.left() or 1000000
    for i in range(round):
        df, review = story.move_forward(df)
        print(review)
        chart(df.copy(), sp.symbol.name, axlist)
        # plt.draw()  # 绘制更新
        plt.pause(0.01)  # 暂停1秒以确保更新完成

    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终图表
    return 


    # # preparing the data
    # y = [random.randint(1,10) for i in range(20)]
    # x = [*range(1,21)]

    # # plotting the first frame
    # graph = plt.plot(x,y)[0]
    # plt.ylim(0,10)
    # plt.pause(1)

    # # the update loop
    # while(True):
    #     # updating the data
    #     y.append(random.randint(1,10))
    #     x.append(x[-1]+1)
        
    #     # removing the older graph
    #     graph.remove()
        
    #     # plotting newer graph
    #     graph = plt.plot(x,y,color = 'g')[0]
    #     plt.xlim(x[0], x[-1])
        
    #     # calling pause function for 0.25 seconds
    #     plt.pause(0.25)


if __name__ == "__main__":
    main(StrategyParam(**params))
    # start_journey(StrategyParam(**params))