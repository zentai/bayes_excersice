import sys, os

from huobi.client.algo import AlgoClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import logging
import time
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

from huobi.client.trade import TradeClient
from huobi.client.account import AccountClient
from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.utils import *

logger = logging.getLogger(__name__)


huobi_interval = {
    "1min": CandlestickInterval.MIN1,
    "5min": CandlestickInterval.MIN5,
    "15min": CandlestickInterval.MIN15,
    "30min": CandlestickInterval.MIN30,
    "60min": CandlestickInterval.MIN60,
    "4hour": CandlestickInterval.HOUR4,
    "1day": CandlestickInterval.DAY1,
    "1mon": CandlestickInterval.MON1,
    "1week": CandlestickInterval.WEEK1,
    "1year": CandlestickInterval.YEAR1,
}


@dataclass
class Candlestick:
    Date: int
    High: float
    Low: float
    Open: float
    Close: float
    Amount: float
    Count: int
    Vol: float

    def __post_init__(self):
        self.Date = datetime.fromtimestamp(self.Date)


# Account
def get_spot_acc(api_key, secret_key):
    account_client = AccountClient(api_key=api_key, secret_key=secret_key)
    return account_client.get_account_by_type_and_symbol(
        account_type=AccountType.SPOT, symbol=None
    )


def get_balance(
    symbol, api_key, secret_key, balance_type=AccountBalanceUpdateType.TRADE
):
    account_client = AccountClient(api_key=api_key, secret_key=secret_key)
    spot_account_id = get_spot_acc(api_key, secret_key).id
    list_obj = account_client.get_balance(spot_account_id)
    symbol = symbol.replace("usdt", "") or "usdt"
    for obj in list_obj.list:
        if obj.currency != symbol:
            continue
        elif obj.type != balance_type:
            continue
        return float(obj.balance)


def load_orders(api_key, secret_key, symbol, order_ids=[], start_deal=None):
    try:
        trade_client = TradeClient(api_key=api_key, secret_key=secret_key)
        cols = [
            "Time",
            "id",
            "symbol",
            "type",
            "state",
            "filled_amount",
            "filled_fees",
            "filled_cash_amount",
            "source",
            "client_order_id",
            "finished_timestamp",
        ]

        orderlist = [
            trade_client.get_order(order_id=order_id) for order_id in order_ids
        ] or trade_client.get_orders(
            symbol=symbol,
            order_state=OrderState.FILLED,
            direct=QueryDirection.PREV,
            start_id=start_deal,
        )

        orders = []
        for order in orderlist:
            finished_at = datetime.fromtimestamp(order.finished_at / 1000)
            orders.append(
                [
                    finished_at,
                    order.id,
                    order.symbol,
                    order.type,
                    order.state,
                    float(order.filled_amount),
                    float(order.filled_fees),
                    float(order.filled_cash_amount),
                    order.source,
                    order.client_order_id,
                    order.finished_at,
                ]
            )

        orders = pd.DataFrame(orders, columns=cols)
        orders = orders.sort_values(by=["Time"])

        # Define order type constants
        buy_types = ("buy-stop-limit", "buy-limit", "buy-market")
        sell_types = ("sell-limit", "sell-stop-limit", "sell-market")
        valid_status = ("filled", "partial-canceled", "partial-filled")

        # Filter orders by valid status
        orders = orders[orders.state.isin(valid_status)]

        # Create masks for buy and sell orders
        buy_mask = orders.type.isin(buy_types)
        sell_mask = orders.type.isin(sell_types)

        # 计算买入和卖出订单的 position 和 price
        # 买入订单的 position 减去手续费，卖出订单的 position 不变
        # 所有订单的价格计算为成交金额除以 position
        orders.loc[buy_mask, "position"] = (
            orders.loc[buy_mask, "filled_amount"] - orders.loc[buy_mask, "filled_fees"]
        )
        orders.loc[sell_mask, "position"] = orders.loc[sell_mask, "filled_amount"]
        orders.loc[buy_mask, "price"] = (
            orders["filled_cash_amount"] / orders["position"]
        )
        orders.loc[sell_mask, "price"] = (
            orders["filled_cash_amount"] - orders["filled_fees"]
        ) / orders["position"]

        mask = orders.source == "spot-web"
        # Generate client_order_id for web orders to match trading system format: {symbol}_{timestamp}_CUTOFF
        if mask.any():
            orders.loc[mask, "client_order_id"] = (
                orders.loc[mask, "symbol"]
                + "_"
                + orders.loc[mask, "Time"].dt.strftime("%Y%m%d_%H%M00")
                + "_CUTOFF"
            )
        return orders

    except Exception as e:
        logger.error(f"load_orders: {e}")
        time.sleep(5)
        return load_orders(api_key, secret_key, symbol, order_ids)


def get_open_orders(symbol, api_key, secret_key):
    trade_client = TradeClient(api_key=api_key, secret_key=secret_key)
    spot_account_id = get_spot_acc(api_key, secret_key).id
    result = trade_client.get_open_orders(symbol=symbol, account_id=spot_account_id)
    return result


def place_order(
    symbol,
    amount,
    price,
    order_type,
    api_key,
    secret_key,
    stop_price=None,
    operator=None,
    client_order_id=None,
):
    side = {
        "BL": OrderSide.BUY,
        "B": OrderType.BUY_LIMIT,
        "S": OrderType.SELL_LIMIT,
        "SL": OrderSide.SELL,
        "SM": OrderType.SELL_MARKET,
        "BM": OrderType.BUY_MARKET,
    }
    order_type = side.get(order_type)

    # Adjust precision and round the order amount, price and trigger price based on order type (buy/sell)
    if order_type in (
        OrderType.BUY_LIMIT,
        OrderSide.BUY,
        OrderType.BUY_MARKET,
    ):
        amount = (
            symbol.round_price(amount)
            if order_type == OrderType.BUY_MARKET
            else symbol.round_amount(amount)
        )
        price = symbol.round_price(price)
        stop_price = symbol.round_price(stop_price) if stop_price else None

    elif order_type in (OrderType.SELL_LIMIT, OrderSide.SELL):
        amount = symbol.round_amount(amount)
        price = symbol.round_price(price)
        stop_price = symbol.round_price(stop_price) if stop_price else None

    # Place order for BUY_LIMIT (BL) and SELL_LIMIT (SL) order types
    if order_type in (OrderSide.BUY):
        algo_client = AlgoClient(api_key=api_key, secret_key=secret_key)
        spot_account_id = get_spot_acc(api_key, secret_key).id
        order_id = algo_client.create_order(
            symbol=symbol.name,
            account_id=spot_account_id,
            order_side=order_type,
            order_type=AlgoOrderType.LIMIT,
            order_size=f"{amount:.{symbol.amount_prec}f}",
            order_price=f"{price:.{symbol.price_prec}f}",
            stop_price=f"{stop_price:.{symbol.price_prec}f}",
            client_order_id=client_order_id,
        )
    else:
        trade_client = TradeClient(api_key=api_key, secret_key=secret_key)
        spot_account_id = get_spot_acc(api_key, secret_key).id
        order_id = trade_client.create_order(
            symbol=symbol.name,
            account_id=spot_account_id,
            order_type=order_type,
            source=OrderSource.API,
            amount=f"{amount:.{symbol.amount_prec}f}",
            price=f"{price:.{symbol.price_prec}f}",
            # stop_price=f"{stop_price:.{symbol.price_prec}f}",
            # operator=operator,
            client_order_id=client_order_id,
        )

    print(
        f"[huobi_api] - {client_order_id} placed, Price: {price}, Amount: {amount}, Stop Price: {stop_price if stop_price else 'N/A'}"
    )
    return order_id


def cancel_algo_open_orders(api_key, secret_key, symbol, isBuy):
    algo_client = AlgoClient(api_key=api_key, secret_key=secret_key)
    trade_client = TradeClient(api_key=api_key, secret_key=secret_key)

    spot_account_id = get_spot_acc(api_key, secret_key).id

    success = []
    if isBuy:

        for o in algo_client.get_open_orders() or []:
            if "BUY" in o.clientOrderId:
                result = algo_client.cancel_orders([o.clientOrderId])
                success += result.accepted

        for o in trade_client.get_open_orders(
            symbol=symbol, account_id=spot_account_id, direct=QueryDirection.PREV
        ):
            if "BUY" in o.client_order_id:
                result = trade_client.cancel_client_order(
                    client_order_id=o.client_order_id
                )
                success.append(o.client_order_id)

    else:
        for o in algo_client.get_open_orders() or []:
            if "BUY" not in o.clientOrderId:
                result = algo_client.cancel_orders([o.clientOrderId])
                success += result.accepted

        for o in trade_client.get_open_orders(
            symbol=symbol, account_id=spot_account_id, direct=QueryDirection.PREV
        ):
            if "BUY" not in o.client_order_id:
                result = trade_client.cancel_client_order(
                    client_order_id=o.client_order_id
                )
                success.append(o.client_order_id)

    return success


def cancel_all_open_orders(symbol, api_key, secret_key, order_type=None):
    try:
        trade_client = TradeClient(api_key=api_key, secret_key=secret_key)
        orders = get_open_orders(symbol, api_key, secret_key)
        c_success, c_fail = [], []
        for order in orders:
            if order.source != "api":
                continue
            if order_type and order.type != order_type:
                continue
            logger.info(
                f"Cancel: {order.id}, {order.type}, {order.state}, {order.filled_amount}, {order.filled_cash_amount}"
            )
            canceled_order_id = trade_client.cancel_order(order.symbol, order.id)
            if canceled_order_id == order.id:
                c_success.append(order.id)
            else:
                c_fail.append(order.id)
        return c_success, c_fail
    except Exception as e:
        logger.error(f"Cancel fail, {e}")
        time.sleep(5)
        return cancel_all_open_orders(symbol, api_key, secret_key)


def load_history_orders(symbol, api_key, secret_key, size=1000):
    trade_client = TradeClient(api_key=api_key, secret_key=secret_key)
    return [obj.id for obj in trade_client.get_history_orders(symbol=symbol, size=size)]


# Market
def get_history_stick(symbol, sample=20, interval="1min"):
    interval = huobi_interval.get(interval)
    market_client = MarketClient(init_log=False, timeout=10)
    htx_stick = market_client.get_candlestick(symbol, interval, sample)

    candlesticks = [
        Candlestick(
            stick.id,
            stick.high,
            stick.low,
            stick.open,
            stick.close,
            stick.amount,
            stick.count,
            stick.vol,
        )
        for stick in htx_stick
    ]
    df = pd.DataFrame(candlesticks)
    return df.sort_values(by=["Date"]).reset_index(drop=True)


def get_market_detail_merged(symbol):
    """获取市场最新聚合行情

    Args:
        symbol: 交易对名称

    Returns:
        MarketDetailMerged对象，包含:
        - id: 响应id
        - timestamp: 响应生成时间点，单位毫秒
        - bid: [买1价,买1量]
        - ask: [卖1价,卖1量]
        - open: 24小时开盘价
        - close: 最新价
        - high: 24小时最高价
        - low: 24小时最低价
        - amount: 24小时成交量(币)
        - vol: 24小时成交额(USDT)
        - count: 24小时成交笔数
    """
    market_client = MarketClient()
    obj = market_client.get_market_detail_merged(symbol)
    return obj


def get_strike(symbol):
    df = get_history_stick(symbol, sample=1)
    return df.iloc[len(df.index) - 1].Close


def seek_price(symbol, action):
    market_client = MarketClient()
    seeking = True
    while seeking:
        list_obj = market_client.get_market_trade(symbol=symbol.name)
        for t in list_obj:
            if t.direction == action:
                return t.price
        time.sleep(0.05)


def subscribe_order_update(symbol, api_key, secret_key, callback_func, error_handler):
    trade_client = TradeClient(api_key=api_key, secret_key=secret_key, init_log=False)
    trade_client.sub_order_update(symbol, callback_func, error_handler)


from huobi.client.trade import TradeClient
from huobi.constant import *
from huobi.model.trade import OrderUpdateEvent


def callback(upd_event: "OrderUpdateEvent"):
    print("---- order update : ----")
    upd_event.print_object()
    print()


if __name__ == "__main__":
    # get_balance('nearusdt')
    # get_balance('usdt')

    # order_id = place_order(symbol, 2170.5, 0.004949, 'SL', stop_price=0.005000, operator='gte')
    # succ, fail = cancel_all_open_orders(symbol.name, OrderType.BUY_LIMIT)
    # print(succ)
    # print(fail)
    # print(f"[SEEK BUY] {seek_price(symbol=symbol, action='buy')}")
    # order_id = place_order(symbol=symbol, amount=10, price=None, order_type='BM')
    # df_orders = get_orders([order_id])
    # print(df_orders)
    # price = df_orders.loc[df_orders.id == order_id].filled_cash_amount.iloc[0]
    # position = df_orders.loc[df_orders.id == order_id].filled_amount.iloc[0]
    # print(f"[BUY] Average Price: {price/position} ")
    # time.sleep(1)
    # print(f"[SEEK SELL] {seek_price(symbol=symbol, action='sell')}")
    # order_id = place_order(symbol=symbol, amount=position, price=None, order_type='SM')

    ########################
    #  Re-Testing anything
    ########################

    # 1. test Spot accound id
    # print(get_spot_acc(api_key, secret_key).id)

    # 2. Test Coin balance
    # print(
    #     get_balance(
    #         "pepeusdt",
    #         g_api_key,
    #         g_secret_key,
    #         balance_type=AccountBalanceUpdateType.TRADE,
    #     )
    # )

    # 3. test get Orders.
    # order_ids = [1220351950556858, 1220339501577514, 1218898472054379]
    # orders = get_orders(order_ids, api_key, secret_key)
    # print(orders)

    # 4. test get_open_orders
    # orders = get_open_orders("pepeusdt", api_key, secret_key)
    # print(orders)

    g_api_key = api_key = "fefd13a1-bg2hyw2dfg-440b3c64-576f2"
    g_secret_key = secret_key = "1a437824-042aa429-0beff3ba-03e26"
    # from hunterverse.interface import Symbol

    # symbol = "pepeusdt"

    # trade_client = TradeClient(api_key=g_api_key, secret_key=g_secret_key)
    # account_client = AccountClient(api_key=g_api_key, secret_key=g_secret_key)

    # account_spot = account_client.get_account_by_type_and_symbol(
    #     account_type=AccountType.SPOT, symbol=None
    # )
    # account_id_test = account_spot.id

    # direct_tmp = QueryDirection.NEXT
    # LogInfo.output(
    #     "==============test case 1 for {direct}===============".format(
    #         direct=direct_tmp
    #     )
    # )
    # list_obj = trade_client.get_open_orders(
    #     symbol=symbol, account_id=account_id_test, direct=direct_tmp
    # )
    # LogInfo.output_list(list_obj)

    # direct_tmp = QueryDirection.PREV
    # LogInfo.output(
    #     "==============test case 2 for {direct}===============".format(
    #         direct=direct_tmp
    #     )
    # )
    # list_obj = trade_client.get_open_orders(
    #     symbol=symbol, account_id=account_id_test, direct=direct_tmp
    # )
    # LogInfo.output_list(list_obj)

    # result = trade_client.cancel_client_order(client_order_id="20250204_060000_BUY")
    # print(type(result))
    # LogInfo.output("cancel result {id}".format(id=result))

    # algo_client = AlgoClient(api_key=g_api_key, secret_key=g_secret_key)
    # spot_account_id = get_spot_acc(api_key, secret_key).id
    # try:
    #     result = algo_client.get_open_orders()
    #     # LogInfo.output_list(result)

    #     result = algo_client.cancel_orders([r.clientOrderId for r in result[:20]])
    #     result.accepted
    #     print("Cancel result:")
    #     result.print_object()
    # except Exception as e:
    #     print(e)

    # today = datetime.now()
    # client_order_id = f"{today.strftime('%Y%m%d_%H%M%S')}"
    # price = 0.000018650000
    # stop_price = price + 0.0000001
    # print(f"{client_order_id} => {price:.12f} < {stop_price:.12f}")
    # order_id = algo_client.create_order(
    #     symbol="pepeusdt",
    #     account_id=spot_account_id,
    #     order_side=OrderSide.SELL,
    #     order_type=AlgoOrderType.LIMIT,
    #     order_size="8419228.18",
    #     order_price=f"{price:.12f}",
    #     stop_price=f"{stop_price:.12f}",
    #     client_order_id=client_order_id,
    # )
    # print(f"{client_order_id} => {order_id}")

    # time.sleep(10)
    # result = algo_client.cancel_orders([client_order_id])
    # result.accepted
    # print("Cancel result:")
    # result.print_object()

    # 5. Test Place order
    # order_id = place_order(
    #     Symbol("pepeusdt"),
    #     10004418.48,
    #     0.000018000000,
    #     "B",
    #     api_key,
    #     secret_key,
    #     stop_price=0.000017900000,
    #     operator="gte",
    # )
    # print(order_id)

    # 6. cancel_all_open_orders(symbol, api_key, secret_key, order_type=None)
    # success, fail = cancel_all_open_orders(
    #     "pepeusdt", api_key, secret_key, order_type=None
    # )
    # print(success, fail)

    # 7. load_history_orders(symbol, api_key, secret_key, size=1000):
    # print(load_history_orders("pepeusdt", api_key, secret_key, size=1000))

    # 8. def get_history_stick(symbol, sample=20, interval="1min"):
    # print(get_history_stick("pepeusdt", sample=20, interval="1min"))

    # account_client = AccountClient(api_key=g_api_key, secret_key=g_secret_key)
    # spot_account_id = get_spot_acc(api_key, secret_key).id
    # list_obj = account_client.get_balance(account_id=spot_account_id)
    # LogInfo.output_list(list_obj)
    # from huobi.utils import LogInfo

    # 9.  get_strike(symbol):
    # print(get_strike("pepeusdt"))

    # 10. def seek_price(symbol, action):
    # print(seek_price(Symbol("pepeusdt"), "buy"))

    # trade_client = TradeClient(api_key=api_key, secret_key=secret_key, init_log=False)
    # trade_client.sub_order_update("pepeusdt", callback)
    # time.sleep(10)

    # account_client = AccountClient(api_key=g_api_key, secret_key=g_secret_key)
    # LogInfo.output(
    #     "====== (SDK encapsulated api) not recommend for low performance and frequence limitation ======"
    # )
    # account_balance_list = account_client.get_account_balance()
    # if account_balance_list and len(account_balance_list):
    #     for account_obj in account_balance_list:
    #         print(account_obj)
    #         account_obj.print_object()
    #         print()

    # df_orders = get_orders([1296264021935932], api_key, secret_key)
    # print(df_orders)

    # symbol = "buzzusdt"
    # trade_client = TradeClient(api_key=g_api_key, secret_key=g_secret_key)
    # list_obj = trade_client.get_orders(
    #     symbol=symbol,
    #     order_state=OrderState.FILLED,
    #     # order_type=OrderType.BUY_LIMIT,
    #     start_date=None,
    #     end_date=None,
    #     start_id=None,
    #     size=None,
    #     direct=QueryDirection.PREV,
    # )
    # LogInfo.output(
    #     "===== step 1 ==== {symbol} {count} orders found".format(
    #         symbol=symbol, count=len(list_obj)
    #     )
    # )
    # LogInfo.output_list(list_obj)
    # price = df_orders.loc[df_orders.id == order_id].filled_cash_amount.iloc[0]
    # position = df_orders.loc[df_orders.id == order_id].filled_amount.iloc[0]
    # print(f"[SELL] Average Price: {price/position} ")

    from huobi.client.market import MarketClient

    market_client = MarketClient()
    obj = market_client.get_market_detail_merged("btcusdt")
    print(obj.vol)
    # obj.print_object()
