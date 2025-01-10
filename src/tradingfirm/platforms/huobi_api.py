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


def get_orders(order_ids, api_key, secret_key):
    try:
        trade_client = TradeClient(api_key=api_key, secret_key=secret_key)
        """
            order.finished_at,
            order.id,
            order.symbol,
            order.type,.    # BUY_LIMIT, SELL_LIMIT
            order.state,    # OrderState.FILLED
            order.filled_amount,
            order.filled_fees,
            order.filled_cash_amount,
            order.source,
        """
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
        ]
        orders = []

        for order_id in order_ids:
            order = trade_client.get_order(order_id=order_id)
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
                ]
            )
        orders = pd.DataFrame(orders, columns=cols)
        return orders.sort_values(by=["Time"])

    except Exception as e:
        logger.error(f"get_orders: {e}")
        time.sleep(5)
        return get_orders(order_ids, api_key, secret_key)


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
    if order_type in (
        OrderType.BUY_LIMIT,
        OrderSide.BUY,
        OrderType.BUY_MARKET,
    ):
        round_amount = (
            symbol.round_price(amount)
            if order_type == OrderType.BUY_MARKET
            else symbol.round_amount(amount)
        )
        round_price = symbol.round_price(price)
        round_stop_price = symbol.round_price(stop_price) if stop_price else None
        amount = round_amount
        price = round_price
        stop_price = round_stop_price
        print(
            f"[{order_type}] adjust buy amount: {round_amount}, trigger Price: {round_stop_price}, price: {round_price}"
        )
    elif order_type in (OrderType.SELL_LIMIT, OrderSide.SELL):
        round_amount = symbol.round_amount(amount)
        round_price = symbol.round_price(price)
        stop_price = symbol.round_price(stop_price) if stop_price else None
        amount = round_amount
        price = round_price
        print(
            f"[{order_type}] adjust sell amount: {round_amount}, trigger Price: {stop_price}, price: {round_price}"
        )

    # BL, SL
    if order_type in (OrderSide.BUY, OrderSide.SELL):
        algo_client = AlgoClient(api_key=g_api_key, secret_key=g_secret_key)
        spot_account_id = get_spot_acc(api_key, secret_key).id
        order_id = algo_client.create_order(
            symbol=symbol.name,
            account_id=spot_account_id,
            order_side=order_type,
            order_type=AlgoOrderType.LIMIT,
            order_size=f"{amount}",
            order_price=f"{price}",
            stop_price=f"{stop_price}",
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
            amount=amount,
            price=price,
            stop_price=stop_price,
            operator=operator,
            client_order_id=client_order_id,
        )

    logger.debug(f"[{order_type}] Order placed: {order_id}")
    return order_id


def cancel_algo_open_orders(api_key, secret_key, orders_to_cancel):
    algo_client = AlgoClient(api_key=g_api_key, secret_key=g_secret_key)
    result = algo_client.cancel_orders(orders_to_cancel)
    return result.accepted, result.rejected


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


def subscribe_order_update(symbol, api_key, secret_key, callback_func):
    trade_client = TradeClient(api_key=api_key, secret_key=secret_key, init_log=False)
    trade_client.sub_order_update(symbol, callback_func)


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
    from hunterverse.interface import Symbol

    algo_client = AlgoClient(api_key=g_api_key, secret_key=g_secret_key)
    spot_account_id = get_spot_acc(api_key, secret_key).id

    today = datetime.now()
    client_order_id = f"{today.strftime('%Y%m%d_%H%M%S')}"
    price = 0.000018650000
    stop_price = price + 0.0000001
    print(f"{client_order_id} => {price:.12f} < {stop_price:.12f}")
    order_id = algo_client.create_order(
        symbol="pepeusdt",
        account_id=spot_account_id,
        order_side=OrderSide.SELL,
        order_type=AlgoOrderType.LIMIT,
        order_size="8419228.18",
        order_price=f"{price:.12f}",
        stop_price=f"{stop_price:.12f}",
        client_order_id=client_order_id,
    )
    print(f"{client_order_id} => {order_id}")

    time.sleep(10)
    result = algo_client.cancel_orders([client_order_id])
    result.accepted
    print("Cancel result:")
    result.print_object()

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

    # df_orders = get_orders([1220351950556858], api_key, secret_key)
    # print(df_orders)

    # price = df_orders.loc[df_orders.id == order_id].filled_cash_amount.iloc[0]
    # position = df_orders.loc[df_orders.id == order_id].filled_amount.iloc[0]
    # print(f"[SELL] Average Price: {price/position} ")
