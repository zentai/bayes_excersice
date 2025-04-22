from huobi.client.trade import TradeClient
from huobi.constant import *
from huobi.utils import *

symbol = "pepeusdt"
g_api_key = "fefd13a1-bg2hyw2dfg-440b3c64-576f2"
g_secret_key = "1a437824-042aa429-0beff3ba-03e26"
trade_client = TradeClient(api_key=g_api_key, secret_key=g_secret_key)
list_obj = trade_client.get_orders(
    symbol=symbol,
    order_state=OrderState.FILLED,
    order_type=OrderType.BUY_LIMIT,
    start_date=None,
    end_date=None,
    start_id=None,
    size=None,
    direct=QueryDirection.PREV,
)

LogInfo.output(
    "===== step 1 ==== {symbol} {count} orders found".format(
        symbol=symbol, count=len(list_obj)
    )
)
LogInfo.output_list(list_obj)

symbol = "eosusdt"
list_obj = trade_client.get_orders(
    symbol=symbol,
    order_state=OrderState.CANCELED,
    order_type=OrderType.BUY_LIMIT,
    start_date="2020-05-21",
    end_date=None,
    start_id=None,
    size=None,
    direct=QueryDirection.PREV,
)
LogInfo.output(
    "===== step 2 ==== {symbol} {count} canceled buy limit orders found".format(
        symbol=symbol, count=len(list_obj)
    )
)
LogInfo.output_list(list_obj)

list_obj = trade_client.get_orders(
    symbol=symbol,
    order_state=OrderState.FILLED,
    order_type=None,
    start_date=None,
    end_date=None,
    start_id=None,
    size=None,
    direct=QueryDirection.PREV,
)
LogInfo.output(
    "===== step 3 ==== {symbol} {count} filled orders found".format(
        symbol=symbol, count=len(list_obj)
    )
)
LogInfo.output_list(list_obj)
