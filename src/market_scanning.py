import sys, os
import click
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from huobi.client.market import MarketClient
from huobi.client.generic import GenericClient
from huobi.utils import *
from huobi.constant import *
from .utils import pandas_util
from .hunterverse.interface import Symbol
from config import config
from .story import DEBUG_COL, params
from .sensor.market_sensor import HuobiMarketSensor, YahooMarketSensor
from .strategy.turtle_trading import TurtleScout
from .hunterverse.interface import StrategyParam
DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir

from .story import entry

stock_watching_list = {
    # UEM
    "5148.KL": "UEM Sunrise Berhad",
    "8583.KL": "Mah Sing Group Berhad",         # check
    "8567.KL": "Eco World Development Group Berhad",  # check
    "5299.KL": "S P Setia Berhad",
    "3743.KL": "IOI Properties Group Berhad", # check
    "3336.KL": "Sunway Berhad",
    "5202.KL": "I-Berhad",
    "5053.KL": "OSK Holdings Berhad",
    "7164.KL": "LBS Bina Group Berhad",
    "5205.KL": "Matrix Concepts Holdings Berhad",
    "1724.KL": "Paramount Corporation Berhad",
    "1171.KL": "Crescendo Corporation Berhad",
    "7123.KL": "Tambun Indah Land Berhad",
    "5184.KL": "WCT Holdings Berhad",
    "3417.KL": "Eastern & Oriental Berhad",
    "5182.KL": "Avaland Berhad",
    "0166.KL": "Inari Amertron Berhad",
    "5008.KL": "Tropicana Corporation Berhad",
    "5200.KL": "UOA Development Bhd",
    "5038.KL": "KSL Holdings Berhad",
    "5239.KL": "Titijaya Land Berhad",
    "5075.KL": "Plenitude Berhad",
    "5175.KL": "Ivory Properties Group Berhad",
    "5020.KL": "Glomac Berhad",
    "1651.KL": "Malaysian Resources Corporation Berhad",
    "5062.KL": "Hua Yang Berhad",
    "6076.KL": "Encorp Berhad", 

    # YTL
    "4677.KL": "YTL Corporation Berhad",
    "6742.KL": "YTL Power International Berhad",
    "5109.KL": "YTL Hospitality REIT",
    "P40U.SI": "Starhill Global REIT",

    # ZEN List
    "GRAB": "GRAB holdings",
    "5318.KL": "DXN holdings",
    "1023.KL": "CIMB",

    # OTher's maybe
    "5183.KL": "Petronas Chemicals Group",
    "5347.KL": "Tenaga Nasional Berhad",
    "7293.KL": "Yinson Holdings Berhad",
    "7033.KL": "Dialog Group Berhad",
    "5196.KL": "Bumi Armada Berhad",
    "5279.KL": "Serba Dinamik Holdings Berhad",
    "5204.KL": "Petronas Gas Berhad",
    "5209.KL": "Gas Malaysia Berhad",
    "2445.KL": "IOI Corporation Berhad",
    "5185.KL": "Sime Darby Plantation Berhad",
    "3476.KL": "MMC Corporation Berhad",
    "2828.KL": "IJM Corporation Berhad",
    "0018.KL": "GHL Systems Berhad",
    "0023.KL": "Datasonic Group Berhad",
    "0138.KL": "MyEG Services Berhad",
    "0041.KL": "Revenue Group Berhad",
    "0082.KL": "Green Packet Berhad",
    "0216.KL": "Vitrox Corporation Berhad",
    "0072.KL": "Inari Amertron Berhad",
    "1155.KL": "Malayan Banking Berhad (Maybank)",
    "1295.KL": "Public Bank Berhad",
    "6888.KL": "RHB Bank Berhad",
    "5819.KL": "Hong Leong Bank Berhad",
    "0146.KL": "Dataprep Holdings Berhad",
    "7183.KL": "IHH Healthcare Berhad",
    "5285.KL": "Frontken Corporation Berhad",
    "3867.KL": "Axiata Group Berhad",
    "6012.KL": "Maxis Berhad",

    # 工业
    "7113.KL": "Top Glove Corporation Berhad",     # 工业产品
    "5168.KL": "Hartalega Holdings Berhad",        # 工业产品
    "7153.KL": "Kossan Rubber Industries Berhad",  # 工业产品
    "7106.KL": "Supermax Corporation Berhad",      # 工业产品

    # 材料
    "8869.KL": "Press Metal Aluminium Holdings Berhad",  # 金属
    "6556.KL": "Ann Joo Resources Berhad",               # 钢铁
    "3794.KL": "Malayan Cement Berhad",                  # 建材
    "4065.KL": "PPB Group Berhad",                       # 多元化材料

    # 必需消费品
    "4707.KL": "Nestle (Malaysia) Berhad",               # 食品
    "3689.KL": "Fraser & Neave Holdings Bhd",            # 饮料
    "7084.KL": "QL Resources Berhad",                    # 农产品
    "3026.KL": "Dutch Lady Milk Industries Berhad",      # 乳制品
    "7216.KL": "Kawan Food Berhad",                      # 食品

    # 非必需消费品
    "4715.KL": "Genting Malaysia Berhad",                # 娱乐
    "1562.KL": "Berjaya Sports Toto Berhad",             # 娱乐
    "6947.KL": "Digi.Com Berhad",                        # 通信
    "1066.KL": "RHB Bank Berhad",                        # 银行

    # 公用事业
    "5264.KL": "MALAKOFF CORPORATION BERHAD",            # 电力

    # 运输
    "5246.KL": "Westports Holdings Berhad",              # 港口运营
    "5014.KL": "Malaysia Airports Holdings Berhad",      # 机场运营
    "3816.KL": "MISC Berhad",                            # 航运
    "5099.KL": "AirAsia Group Berhad",                   # 航空
    "0078.KL": "GDEX Berhad",                            # 物流

}


def fast_scanning():
    market_client = MarketClient(init_log=True)
    list_obj = market_client.get_market_tickers()
    symbols = []
    for obj in list_obj:
        amount = obj.amount
        count = obj.count
        open = obj.open
        close = obj.close
        low = obj.low
        high = obj.high
        vol = obj.vol
        symbol = obj.symbol
        bid = obj.bid
        bidSize = obj.bidSize
        ask = obj.ask
        askSize = obj.askSize
        if (amount * close >= 10000000):
            symbols.append(symbol)
    return symbols

def count_obv_cross(IMarketSensorImp, ccy, interval, sample, show=False):
    params.update({
        "interval": interval,
        "funds": 100,
        "stake_cap": 50,
        "symbol": Symbol(ccy),
    })
    sensor = IMarketSensorImp(symbol=params.get("symbol"), interval=interval)
    df = sensor.scan(sample)
    df = sensor.fetch(df)
    df.to_csv(f"{DATA_DIR}/{ccy}_cached.csv", index=True)
    print(f"{DATA_DIR}/{ccy}_cached.csv")
    sp = StrategyParam(**params)
    scout = TurtleScout(params=sp)
    df = scout._calc_OBV(df, multiplier=3)
    if df.iloc[-1].OBV_UP:
        print("============= OBV UP ============")
    print(f"{ccy=}: {len(df[df.OBV_UP])}")
    if show and df[-90:].OBV_UP.any():
        chart(df[-90:], ccy)
    return len(df[df.OBV_UP])


def chart(df, code):
    # Ensure the index is of type DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
    
    # Create columns for scatter plots with NaN where there's no marker
    obv_up_marker = np.where(df["OBV_UP"], df["OBV"], np.nan)
    close_up_marker = np.where(df["OBV_UP"], df["Close"], np.nan)
    
    # Prepare additional plots
    add_plots = [
        mpf.make_addplot(df["OBV"], panel=1, color='g', secondary_y=False, ylabel='OBV'),
        mpf.make_addplot(df["upper_bound"], panel=1, color='gray', linestyle='--'),
        mpf.make_addplot(df["lower_bound"], panel=1, color='gray', linestyle='--'),
        mpf.make_addplot(obv_up_marker, panel=1, type='scatter', markersize=100, marker='^', color='red'),
        mpf.make_addplot(close_up_marker, type='scatter', markersize=100, marker='^', color='y'),
    ]

    # Create a candlestick chart with additional plots
    mpf.plot(df, type='candle', addplot=add_plots, title=f"{stock_watching_list.get(code, code)} Price and OBV with Bounds", ylabel='Price (USD)', style='yahoo', datetime_format='%Y-%m-%d %H:%M:%S')


@click.command()
@click.option("--ccy", default="bomeusdt", required=False, help="trade ccy pair")
@click.option(
    "--interval",
    required=False,
    default="1day",
    help="trade interval: 1min 5min 15min 30min 60min 4hour 1day 1mon 1week 1year",
)
@click.option(
    "--show",
    required=False,
    default=False,
    help="show chart",
)
def main(ccy, interval, show):
    if ccy == "watching":
        sensor_cls = YahooMarketSensor
        symbols = stock_watching_list.keys()
        sample = 365*20 # 20 years
    else:
        sensor_cls = HuobiMarketSensor
        symbols = fast_scanning()
        sample = 240*3 #

    symbols_count = {s: count_obv_cross(sensor_cls, s, interval, sample, show) for s in symbols}
    from collections import Counter
    c = Counter(symbols_count)
    from pprint import pprint
    pprint(c.most_common()) 


if __name__ == "__main__":
    result = {code: entry(code, "1day", 100, 50) for code in stock_watching_list.keys()}
    from pprint import pprint 
    pprint(result)
    # main()
