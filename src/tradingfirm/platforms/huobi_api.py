import logging
import time
from datetime import datetime
import pandas as pd
from market.base import Candlestick

import huobi
from huobi.client.trade import TradeClient
from huobi.client.account import AccountClient
from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.utils import *

logger = logging.getLogger(__name__)

g_api_key = "uymylwhfeg-eb0f1107-98ea054c-ad39b"
g_secret_key = "6fc09731-0b3fde88-b83e1f10-2dc2f"

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

# Account
def get_spot_acc():
    account_client = AccountClient(api_key=g_api_key, secret_key=g_secret_key)
    return account_client.get_account_by_type_and_symbol(
        account_type=AccountType.SPOT, symbol=None
    )


def get_balance(symbol, balance_type=AccountBalanceUpdateType.TRADE):
    account_client = AccountClient(api_key=g_api_key, secret_key=g_secret_key)
    list_obj = account_client.get_balance(account_id=spot_account_id)
    symbol = symbol.replace("usdt", "") or "usdt"
    for obj in list_obj:
        if obj.currency != symbol:
            continue
        elif obj.type != balance_type:
            continue
        return float(obj.balance)


spot_account_id = get_spot_acc().id


# Trade
def get_orders(order_ids):
    try:
        trade_client = TradeClient(api_key=g_api_key, secret_key=g_secret_key)
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
        return get_orders(order_ids)


def get_open_orders(symbol):
    trade_client = TradeClient(api_key=g_api_key, secret_key=g_secret_key)
    # result = trade_client.get_order(order_id=421444228622801)
    result = trade_client.get_open_orders(symbol=symbol, account_id=spot_account_id)
    return result


def place_order(symbol, amount, price, order_type):
    side = {
        "B": OrderType.BUY_LIMIT,
        "S": OrderType.SELL_LIMIT,
        "SM": OrderType.SELL_MARKET,
    }
    order_type = side.get(order_type)
    trade_client = TradeClient(api_key=g_api_key, secret_key=g_secret_key)
    order_id = trade_client.create_order(
        symbol=symbol,
        account_id=spot_account_id,
        order_type=order_type,
        source=OrderSource.API,
        amount=amount,
        price=price,
    )
    logger.debug(f"[{order_type}] Order placed: {order_id}")
    return order_id


def cancel_all_open_orders(symbol):
    try:
        trade_client = TradeClient(api_key=g_api_key, secret_key=g_secret_key)
        orders = get_open_orders(symbol)
        c_success, c_fail = [], []
        for order in orders:
            if order.source != "api":
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
        return cancel_all_open_orders(symbol)


# Market
def get_history_stick(symbol, sample=20, interval="1min"):
    interval = huobi_interval.get(interval)
    market_client = MarketClient(init_log=True, timeout=10)
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


if __name__ == "__main__":
    # get_balance('nearusdt')
    # get_balance('usdt')
    get_spot_acc().print_object()
    balance = get_balance("usdt")
    print(balance)
