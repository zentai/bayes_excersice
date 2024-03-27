import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from huobi.client.market import MarketClient
from huobi.client.generic import GenericClient
from huobi.utils import *
from huobi.constant import *
from utils import pandas_util
from hunterverse.interface import Symbol

# generic_client = GenericClient()
# list_obj = generic_client.get_exchange_symbols()
# if len(list_obj):
#     for idx, row in enumerate(list_obj):
#         # row.print_object()
#         symbol = row.symbol
#         leverage_ratio = row.leverage_ratio
#         state = row.state
#         max_order_value = row.max_order_value
#         print(f"[{symbol}] {leverage_ratio} {max_order_value}")


market_client = MarketClient(init_log=True)
list_obj = market_client.get_market_tickers()
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
    if amount * close >= 1000000 and close < 0.001 and vol > 100000:
        new_data = pandas_util.get_history_stick(
            Symbol(symbol), sample=30, interval=CandlestickInterval.DAY1
        )
        grow = new_data.iloc[-2].Vol / new_data.iloc[0].Vol
        if grow > 2:
            print(f"[{symbol}] {amount} - {close} - {grow}")


# market_client = MarketClient()
# obj = market_client.get_market_detail_merged("btcusdt")
# obj.print_object()


# obj = market_client.get_market_detail("btcusdt")
# obj.print_object()
# print(type(obj))
# obj.close = 0.0
# obj.amount = 0.0
