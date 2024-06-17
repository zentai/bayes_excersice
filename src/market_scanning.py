import sys, os
import click
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from collections import Counter
import seaborn as sns

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

stocks_singapore = {
    "C6L.SI": "Singapore Airlines - Airline",
    "U96.SI": "Sembcorp Industries - Utilities",
    "D05.SI": "DBS Group Holdings - Banking",
    "Z74.SI": "Singtel - Telecommunications",
    "C07.SI": "Jardine Cycle & Carriage - Automotive",
    "BN4.SI": "Keppel Corporation - Conglomerate",
    "F34.SI": "Wilmar International - Agriculture",
    "BS6.SI": "Yangzijiang Shipbuilding - Shipbuilding",
    "O39.SI": "OCBC Bank - Banking",
    "9CI.SI": "CapitaLand - Real Estate",
    "U11.SI": "United Overseas Bank - Banking",
    "N2IU.SI": "Mapletree Commercial Trust - Real Estate",
    "H78.SI": "Hongkong Land Holdings - Real Estate",
    "S68.SI": "Singapore Exchange - Financial Services",
    "A17U.SI": "Ascendas REIT - Real Estate",
    "C09.SI": "City Developments - Real Estate",
    "M44U.SI": "Mapletree Logistics Trust - Real Estate",
    "V03.SI": "Venture Corporation - Electronics",
    "J69U.SI": "Frasers Logistics & Commercial Trust - Real Estate",
    "ME8U.SI": "Mapletree Industrial Trust - Real Estate",
    "AJBU.SI": "Ascott Residence Trust - Hospitality",
    "NOBGY": "Noble Group - Commodities",
    "K71U.SI": "Keppel REIT - Real Estate",
    "G13.SI": "Genting Singapore - Hospitality",
    "Y92.SI": "Thai Beverage - Beverages",
    "Q5T.SI": "Far East Hospitallity Trust - REITS",
    "U14.SI": "UOL Group - Real Estate",
    "S63.SI": "ST Engineering - Engineering",
    "NS8U.SI": "Hutchison Port Holdings Trust  - REITS",
    "C52.SI": "ComfortDelGro - Transportation",
    "H02.SI": "Haw Par Corporation Limited ",
    "E5H.SI": "Golden Agri-Resources - Agriculture",
    "S59.SI": "SIA Engineering - Aerospace",
    "U09.SI": "Avarga Limited",
    "O5RU.SI": "AIMS APAC REIT - Real Estate",
    "BUOU.SI": "Frasers Logistics & Commercial Trust  - Real Estate",
    "M1GU.SI": "Sabana Industrial Real Estate Investment Trust - Real Estate",
    "BVA.SI": "Top Glove Corporation Bhd.",
    "D03.SI": "Del Monte Pacific - Food & Beverage",
    "P40U.SI": "Starhill Global Real Estate Investment Trust ",
    "F9D.SI": "Boustead Singapore Limited",
    "T82U.SI": "Suntec REIT - Real Estate",
    "TQ5.SI": "Frasers Property - Real Estate",
    "BSL.SI": "Raffles Medical Group - Healthcare",
    "1D4.SI": "Aoxin Q & M Dental Group Limited",
    "NR7.SI": "Raffles Education Corporation - Education",
    "5CP.SI": "Silverlake Axis - Technology",
    "T14.SI": "Tianjin Pharmaceutical Da Ren Tang Group Corporation Limited", 
    "C2PU.SI": "Parkway Life Real Estate Investment Trust ",  
    "U96.SI": "Sembcorp Industries - Energy",
    "5TP.SI": "CNMC Goldmine Holdings Limited", 
    "G07.SI": "Great Eastern - Insurance",
    "TQ5.SI": "Frasers Property Limited ", 
    "CC3.SI": "StarHub - Telecommunications",
    "BN4.SI": "Keppel Corporation - Industrial Conglomerate",
    "C09.SI": "City Developments Limited - Real Estate",
    "O10.SI": "Far East Orchard - Real Estate",
    "S58.SI": "SATS Ltd. - Aviation",
    "Y06.SI": "Green Build Technology Limited",
    "Z25.SI": "Yanlord Land Group - Real Estate",
    "Z59.SI": "Yoma Strategic Holdings - Conglomerate",
    "C76.SI": "Creative Technology - Technology",
    "A50.SI": "Thomson Medical Group - Healthcare",
    "5IG.SI": "Gallant Venture Ltd", 
    "1C0.SI": "Emerging Towns & Cities Singapore Ltd.", 
    "C38U.SI": "CapitaLand Integrated Commercial Trust - Real Stack",
    "C52.SI": "ComfortDelGro Corporation - Transportation",
    "G92.SI": "China Aviation Oil (Singapore) Corporation Ltd", 
    "CRPU.SI": "Sasseur REIT - Real Estate",
    "H30.SI": "Hong Fok Corporation Limited", 
    "H13.SI": "Ho Bee Land - Real Estate",
    "1B1.SI": "HC Surgical Specialists - Healthcare",
    "AWX.SI": "AEM Holdings Ltd.", 
    "1J5.SI": "Hyphens Pharma - Pharmaceuticals",
    "B61.SI": "Bukit Sembawang Estates - Real Estate",
    "A7RU.SI": "Keppel Infrastructure Trust", 
    "BTOU.SI": "Manulife US REIT - Real Estate",
    "BDA.SI": "PNE Industries Ltd",
    "9I7.SI": "No Signboard Holdings - Food & Beverage",
    "BWCU.SI": "EC World Real Estate Investment Trust ", 
    "S7OU.SI": "Asian Pay Television Trust", 
    "TS0U.SI": "OUE Commercial REIT - Real Estate",
    "U9E.SI": "China Everbright Water Limited", 
    "S8N.SG": "Sembcorp Marine - Marine",
    "5GZ.SI": "HGH Holdings Ltd.", 
    "RE4.SI": "Geo Energy Resources - Energy",
    "40T.SI": "ISEC Healthcare Ltd.", 
    "U77.SI": "Sarine Technologies - Technology",
    "AJ2.SI": "Ouhua Energy Holdings Limited", 
    "1A4.SI": "AGV Group - Industrial",
    "S41.SI": "Hong Leong Finance Limited", 
    "Q0X.SI": "Ley Choon Group - Construction",
    "S71.SI": "Sunright Limited", 
    "5UX.SI": "Oxley Holdings - Real Estate",
    "5IF.SI": "Natural Cool Holdings Limited - Hospitality",
    "OV8.SI": "Sheng Siong Group - Retail",
    "AIY.SI": "iFast Corporation - Financial Services",
    "5CP.SI": "Silverlake Axis - Technology",
    "P15.SI": "Pacific Century Regional Developments Limited", 
    "5AB.SI": "Trek 2000 International Ltd", 
    "AZI.SI": "AusNet Services - Utilities",
    "U13.SI": "United Overseas Insurance Limited", 
    "558.SI": "UMS Holdings - Semiconductors",
    "1D0.SI": "Kimly Limited", 
    "I07.SI": "ISDN Holdings - Industrial Automation",
    "5UX.SI": "Oxley Holdings Limited", 
    "M35.SI": "Wheelock Properties - Real Estate",
    "A30.SI": "Aspial Corporation Limited", 
    "5G1.SI": "EuroSports Global Limited", 
    "BJZ.SI": "Koda Ltd - Manufacturing",
    "5TT.SI": "Keong Hong Holdings Limited", 
}

stock_malaysia = {
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
        # if (amount * close >= 10000000):
        if (vol >= 5000000):
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
    mpf.plot(df, type='candle', addplot=add_plots, title=f"{stock_malaysia.get(code, code)} Price and OBV with Bounds", ylabel='Price (USD)', style='yahoo', datetime_format='%Y-%m-%d %H:%M:%S')


@click.command()
@click.option("--market", type=click.Choice(['my', 'sg', 'crypto']), required=True, help="选择市场类型")
@click.option("--symbol", default=None, help="指定交易对或股票代码")
@click.option(
    "--interval",
    default="1day",
    help="交易间隔: 1min, 5min, 15min, 30min, 60min, 4hour, 1day, 1mon, 1week, 1year"
)
@click.option(
    "--show",
    is_flag=False,
    help="是否显示图表"
)
@click.option(
    "--backtest",
    is_flag=False,
    help="是否进行回测"
)
def main(market, symbol, interval, show, backtest):
    if market in ("my", "sg"):
        sensor_cls = YahooMarketSensor
        if symbol:
            symbols = [symbol]  
        else: 
            symbols = stock_malaysia.keys() if market == "my" else stocks_singapore.keys()
        sample = 365 * 20  # 20 years
    elif market == "crypto":
        sensor_cls = HuobiMarketSensor
        symbols = [symbol] if symbol else fast_scanning()
        sample = 240 * 3  # 3 years

    symbols_count = {s: count_obv_cross(sensor_cls, s, interval, sample, show) for s in symbols}
    
    if backtest:
        # 进行回测逻辑
        perform_backtest(symbols_count)

    c = Counter(symbols_count)
    from pprint import pprint 
    pprint(c.most_common())

def perform_backtest(symbols_count):
    result = {}
    for code in symbols_count.keys():
        try:
            result[code] = entry(code, "1day", 100, 10.5) 
        except Exception as e:
            print(f"Back test error: {code}, {e}")
    performance_review(result)

def performance_review(backtest_results):
    # 将数据转换为 DataFrame
    df = pd.DataFrame(list(backtest_results.items()), columns=['Stock', 'Return'])
    df.to_csv(f"{REPORTS_DIR}/backtest.csv")
    df = df[df.Return != 0]
    print(df[:60])

    # 总体表现
    average_return = df['Return'].mean()
    total_return = df['Return'].sum()

    # 风险分析
    std_dev = df['Return'].std()
    max_drawdown = df['Return'].min()

    # 胜率分析
    positive_returns = df[df['Return'] > 0].shape[0]
    negative_returns = df[df['Return'] < 0].shape[0]
    win_rate = positive_returns / df.shape[0]
    loss_rate = negative_returns / df.shape[0]

    # Sharpe Ratio
    risk_free_rate = 0.0
    sharpe_ratio = (df['Return'].mean() - risk_free_rate) / df['Return'].std()

    # Sortino Ratio
    negative_returns = df[df['Return'] < 0]['Return']
    downside_std = negative_returns.std()
    sortino_ratio = (df['Return'].mean() - risk_free_rate) / downside_std

    # 打印结果
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")

    # 打印结果
    print(f"平均回报率: {average_return:.2f}%")
    print(f"总回报率: {total_return:.2f}%")
    print(f"回报率标准差: {std_dev:.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"胜率: {win_rate:.2f}")
    print(f"失败率: {loss_rate:.2f}")

    # 绘制收益分布图
    plt.figure(figsize=(14, 8))
    sns.histplot(df['Return'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Returns')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    # result = {code: entry(code, "1day", 100, 50) for code in stock_malaysia.keys()}
    # from pprint import pprint 
    # pprint(result)
    main()
