from icecream import ic
import os
import pandas as pd
import numpy as np
import logging
from huobi.constant.definition import OrderType, OrderState

from ..hunterverse.interface import IHunter
from ..utils import pandas_util
from .platforms import huobi_api

from config import config

ZERO = config.zero

HUNTER_COLUMNS = [
    "xBuy",
    "xSell",
    "xProfit",
    "xPosition",
    "xCash",
    "xAvgCost",
    "xBuyOrder",
    "xSellOrder",
]


def weighted_daily_return(df_subset):
    total_time_cost = df_subset["time_cost"].sum() or 1
    weighted_return = np.sum(
        (1 + df_subset["xProfit"]) ** (1 / df_subset["time_cost"])
        * df_subset["time_cost"]
    )
    wd_return = (weighted_return / total_time_cost) - 1
    return total_time_cost, wd_return


def calc_annual_return_and_sortino_ratio(cost, profit, df):
    if df.empty:
        return 0, 0
    _yield_curve_1yr = 0.0419
    _start_date = df.iloc[0].Date
    _end_date = df.iloc[-1].Date
    _trade_count = len(df[df.Kelly > 0])
    _trade_minutes = (
        pd.to_datetime(_end_date) - pd.to_datetime(_start_date)
    ).total_seconds() / 60 or ZERO
    _annual_trade_count = (_trade_count / _trade_minutes) * 365 * 24 * 60
    _downside_risk_stdv = df[
        (df.Kelly > 0) & (df.xProfit < _yield_curve_1yr)
    ].xProfit.std(ddof=1)
    _annual_downside_risk_stdv = _downside_risk_stdv * np.sqrt(_annual_trade_count)
    t = _trade_minutes / (365 * 24 * 60)
    _annual_return = (profit / cost) ** (1 / t) - 1 if (profit / cost > 0) else 0
    _sortino_ratio = (_annual_return - _yield_curve_1yr) / _annual_downside_risk_stdv
    return _annual_return, _sortino_ratio


class GainsBag:
    MIN_STAKE_CAP = 10.01  # huobi limitation

    def __init__(self, symbol, init_funds, stake_cap, init_position=0, init_avg_cost=0):
        self.logger = logging.getLogger(f"GainsBag_{symbol.name}")
        self.symbol = symbol
        self.stake_cap = stake_cap
        self.init_funds = self.cash = init_funds
        self.position = init_position
        self.avg_cost = init_avg_cost
        self.logger.info(f"Create GainsBag: {self:snapshot}")
        print(f"Create GainsBag: {self:snapshot}")

    def get_un_pnl(self, strike):
        return (strike * self.position) + self.cash - self.init_funds

    def log_action(self, action, price, position, cash):
        price = round(price, self.symbol.price_prec)
        position = round(position, self.symbol.amount_prec)
        cash = round(cash, 2)  # USDT
        unpnl = round(self.get_un_pnl(price), 2)  # USDT
        msg = f"[{self.symbol.name}] ∆ {price} $ {cash} Ⓒ {position} unPNL: {unpnl}"
        if action == "open_position":
            msg = f"+{msg}"
        elif action == "close_position":
            msg = f"-{msg}"
        self.logger.info(msg)

        # dump snapshot
        snapshot_avg_cost = round(self.avg_cost, self.symbol.price_prec)  # USDT
        snapshot_position = round(self.position, self.symbol.amount_prec)
        snapshot_cash = round(self.cash, 2)  # USDT
        print(f"{self:snapshot}")
        self.logger.info(f"{self:snapshot}")

    def discharge(self, ratio):
        stake_amount = self.stake_cap if self.cash > self.stake_cap else self.cash
        if stake_amount < self.MIN_STAKE_CAP:
            self.logger.warning(f"Run out of budget, {self.cash}")
            return 0
        return (
            stake_amount * ratio
            if stake_amount * ratio > self.MIN_STAKE_CAP
            else self.MIN_STAKE_CAP
        )

    def open_position(self, position, price):
        cash = position * price
        self.avg_cost = ((self.avg_cost * self.position) + cash) / (
            self.position + position
        )
        self.cash -= cash
        self.position += position
        self.log_action("open_position", price, position, cash)

    def close_position(self, position, price):
        cash = position * price
        self.cash += cash
        self.position -= position
        self.log_action("close_position", price, position, cash)

    def is_enough_position(self):
        return (self.position * self.avg_cost) > self.MIN_STAKE_CAP

    def is_enough_cash(self):
        return self.cash > self.MIN_STAKE_CAP

    def portfolio(self, pre_strike, strike):
        market_value = self.position * strike
        cost = self.position * self.avg_cost
        return {
            "Pair": self.symbol.name,
            "MarketValue": market_value,
            "BuyValue": cost,
            "ProfitLoss": market_value - cost,
            "ProfitLossPercent": (market_value / cost) - 1,
            "Hr24": (strike / pre_strike) - 1,
            "Position": self.position,
            "AvgCost": self.avg_cost,
            "Strikle": strike,
            "Cash": self.cash,
            "FinalPnL": market_value + self.cash - self.init_funds,
        }

    def __format__(self, format_spec):
        if format_spec == "cash":
            return f"{self.cash}"
        elif format_spec == "avg_cost":
            return f"{self.avg_cost}"
        elif format_spec == "snapshot":
            snapshot_avg_cost = round(self.avg_cost, self.symbol.price_prec)  # USDT
            snapshot_position = round(self.position, self.symbol.amount_prec)
            snapshot_cash = round(self.cash, 2)  # USDT
            return f"![{self.symbol.name}] ∆ {snapshot_avg_cost} $ {snapshot_cash} Ⓒ {snapshot_position}"
        elif format_spec == "review":
            pass
        else:
            return self.name


class xHunter(IHunter):
    def __init__(self, params):
        super().__init__()
        self.buy_types = ("buy-stop-limit", "buy-limit", "buy-market")
        self.sell_types = ("sell-limit", "sell-stop-limit", "sell-market")
        self.status = ("filled", "partial-canceled", "partial-filled")
        self.params = params
        self.fetch_huobi = params.fetch_huobi
        self.simulate = params.simulate
        self.gains_bag = GainsBag(
            symbol=params.symbol, init_funds=params.funds, stake_cap=50
        )
        self.lastest_candlestick = None

    def load_memories(self, fetch=True, deals=[]):
        print(f"load_memories(self, fetch={fetch}, deals={deals})")

        cached_order_ids = []
        if os.path.exists(db_path := f"{config.data_dir}/{self.params.symbol}.csv"):
            cached_order_ids = pd.read_csv(db_path).id.tolist()

        if fetch:
            orders = huobi_api.get_orders(
                set(
                    [ str(i) for i in (cached_order_ids
                    + deals
                    + huobi_api.load_history_orders(f"{self.params.symbol}"))]
                )
            )

        """Process and enrich orders DataFrame with positions and prices."""
        orders = orders[orders.state.isin(self.status)]
        pd.DataFrame(orders).to_csv(db_path, index=False)

        # 分别处理买入和卖出订单
        buy_mask = orders.type.isin(self.buy_types)
        sell_mask = orders.type.isin(self.sell_types)

        # 买入订单的 position 需要减去手续费
        orders.loc[buy_mask, "position"] = (
            orders.loc[buy_mask, "filled_amount"] - orders.loc[buy_mask, "filled_fees"]
        )
        # 卖出订单的 position 不减去手续费
        orders.loc[sell_mask, "position"] = orders.loc[sell_mask, "filled_amount"]

        # 所有订单的价格都是成交金额除以位置大小
        orders.loc[buy_mask, "price"] = (
            orders["filled_cash_amount"] / orders["position"]
        )
        orders.loc[sell_mask, "price"] = (
            orders["filled_cash_amount"] - orders["filled_fees"]
        ) / orders["position"]

        msg = []
        for order in orders.itertuples():
            position, price, filled_cash_amount = (
                order.position,
                order.price,
                order.filled_cash_amount,
            )

            if order.type in self.buy_types:
                self.gains_bag.open_position(position, price)
                print(f"[B] {self.gains_bag:snapshot}")
                msg.append(["-", round(filled_cash_amount, 2), round(price, 8)])
            elif order.type in self.sell_types:
                self.gains_bag.close_position(position, price)
                print(f"[S] {self.gains_bag:snapshot}")
                msg.append(["+", round(filled_cash_amount, 2), round(price, 8)])

        if msg:
            fund = f"{self.gains_bag:cash}"
            cost = float(f"{self.gains_bag:avg_cost}")
            msg.append(["=", fund, cost])
            strike = huobi_api.get_strike(f"{self.params.symbol}")
            market_value = strike - cost
            msg.append(["$", market_value * position, strike])
            msg_df = pd.DataFrame(msg, columns=["@", "USDT", "Price"])
            print(msg_df)

    def strike_phase(self, base_df):
        base_df = pandas_util.equip_fields(base_df, HUNTER_COLUMNS)
        self.lastest_candlestick = lastest_candlestick = base_df.iloc[-1]
        base_df = self.sim_attach(base_df) if self.simulate else self.attack(base_df)
        base_df = self.sim_retreat(base_df) if self.simulate else self.retreat(base_df)
        return base_df

    def sim_attack(self, base_df):
        buy_signal = (
            self.lastest_candlestick.Kelly > 0 and self.gains_bag.is_enough_cash()
        )
        if buy_signal:
            budget = self.gains_bag.discharge(self.lastest_candlestick.Kelly)
            price = self.lastest_candlestick.Close
            if self.fetch_huobi:
                price = huobi_api.seek_price(self.params.symbol, action="buy")
            position = budget / price
            self.gains_bag.open_position(position, price)
            s_buy_order = base_df.Date == self.lastest_candlestick.Date
            base_df.loc[s_buy_order, "xBuy"] = price
            base_df.loc[s_buy_order, "xPosition"] = self.gains_bag.position
            base_df.loc[s_buy_order, "xCash"] = self.gains_bag.cash
            base_df.loc[s_buy_order, "xAvgCost"] = self.gains_bag.avg_cost
        return base_df

    def attack(self, base_df):
        if self.simulate or not self.fetch_huobi:
            return base_df
        # check Limit-buy status
        s_buy_order = (
            base_df.xBuyOrder.notna()
            & (base_df.xBuyOrder != "Cancel")
            & base_df.xBuy.isna()
        )
        if s_buy_order.any():
            order_id = base_df[s_buy_order].xBuyOrder.values
            df_orders = huobi_api.get_orders(order_id)
            if (
                (df_orders.type.isin(self.buy_types))
                & (df_orders.state.isin(self.status))
            ).any():
                cash = df_orders.loc[df_orders.id == order_id].filled_cash_amount.iloc[
                    0
                ]
                position = df_orders.loc[df_orders.id == order_id].filled_amount.iloc[0]
                price = cash / position
                print(f"Order: {order_id} filled, position: {position}  price: {price}")
                self.gains_bag.open_position(position, price)
                print(f"gains_bag: {self.gains_bag:snapshot}")
                base_df.loc[s_buy_order, "xBuy"] = price
                base_df.loc[s_buy_order, "xPosition"] = self.gains_bag.position
                base_df.loc[s_buy_order, "xCash"] = self.gains_bag.cash
                base_df.loc[s_buy_order, "xAvgCost"] = self.gains_bag.avg_cost

        # buy_signal = self.lastest_candlestick.Kelly > 0 and self.gains_bag.is_enough_cash()
        buy_signal = self.gains_bag.is_enough_cash()
        if buy_signal:
            budget = self.gains_bag.discharge(ratio=1)
            # budget = self.gains_bag.discharge(self.lastest_candlestick.Kelly)

            try:
                for order_type in self.buy_types:
                    success, fail = huobi_api.cancel_all_open_orders(
                        self.params.symbol.name, order_type=order_type
                    )
                    base_df.loc[base_df.xBuyOrder.isin(success), "xBuyOrder"] = "Cancel"
            except Exception as e:
                print(f"[xHunter.atack()] Cancel Old trade fail: {e}")

            position = budget / (self.lastest_candlestick.Close * 1.002)
            try:
                price = base_df.tail(self.params.upper_sample).High.max()
                is_fly = price == base_df.iloc[-1].High
                if is_fly:
                    print(f"[B] {price}")
                    order_id = huobi_api.place_order(
                        symbol=self.params.symbol,
                        amount=position,
                        price=price,
                        order_type="B",
                    )
                else:
                    print(f"[BL] {price}")
                    order_id = huobi_api.place_order(
                        symbol=self.params.symbol,
                        amount=position,
                        stop_price=(price * 0.9995),
                        price=price,
                        order_type="BL",
                        operator="gte",
                    )
                s_buy = base_df.Date == self.lastest_candlestick.Date
                base_df.loc[s_buy, "xBuyOrder"] = order_id
            except Exception as e:
                print(f"[xHunter.attack()] place order fail: {e}")
        return base_df

    def sim_retreat(self, base_df):
        cutoff_price = self.params.hard_cutoff * self.gains_bag.avg_cost
        Stop_profit = max(cutoff_price, self.lastest_candlestick.Stop_profit)
        sell_signal = (
            self.lastest_candlestick.Close < Stop_profit
            and self.gains_bag.is_enough_position()
        )
        if sell_signal:
            position = self.gains_bag.position
            price = self.lastest_candlestick.Close
            if self.fetch_huobi:
                price = huobi_api.seek_price(self.params.symbol, action="sell")
            self.gains_bag.close_position(position, price)
            s_sell = base_df.xBuy.notna() & base_df.xSell.isna()
            if s_sell.any():  # should skip all False, mean nothing to update
                last_index = base_df.loc[s_sell].index[-1]
                base_df.loc[s_sell, "xSell"] = price
                base_df.loc[s_sell, "xProfit"] = (base_df.xSell / base_df.xBuy) - 1
                base_df.at[last_index, "xPosition"] = self.gains_bag.position
                base_df.at[last_index, "xCash"] = self.gains_bag.cash
        return base_df

    def retreat(self, base_df):
        if self.simulate or not self.fetch_huobi:
            return base_df
        cutoff_price = self.params.hard_cutoff * self.gains_bag.avg_cost
        cutoff_price = max(cutoff_price, self.lastest_candlestick.exit_price)
        Stop_profit = self.lastest_candlestick.Stop_profit

        pending_order = (
            base_df.xSellOrder.notna()
            & (base_df.xSellOrder != "Cancel")
            & base_df.xSell.isna()
        )
        if pending_order.any():
            order_id = base_df[pending_order].xSellOrder.values
            df_orders = huobi_api.get_orders(order_id)
            if (
                (df_orders.type.isin(self.sell_types))
                & (df_orders.state.isin(self.status))
            ).any():
                cash = df_orders.loc[df_orders.id == order_id].filled_cash_amount.iloc[
                    0
                ]
                position = df_orders.loc[df_orders.id == order_id].filled_amount.iloc[0]
                price = cash / position
                print(f"Order: {order_id} filled, position: {position}  price: {price}")
                self.gains_bag.close_position(position, price)
                print(f"gains_bag: {self.gains_bag:snapshot}")
                base_df.loc[pending_order, "xSell"] = price
                base_df.loc[pending_order, "xProfit"] = (
                    base_df.xSell / base_df.xBuy
                ) - 1

        sell_signal = self.gains_bag.is_enough_position()
        if sell_signal:
            try:
                for order_type in self.sell_types:
                    # reset cutoff order (Sell-Stop-Limit order)
                    success, fail = huobi_api.cancel_all_open_orders(
                        self.params.symbol.name, order_type=order_type
                    )
                    base_df.loc[base_df.xSellOrder.isin(success), "xSellOrder"] = (
                        "Cancel"
                    )
            except Exception as e:
                print(f"[xHunter.retreat()] Cancel Old trade fail: {e}")
            huobi_position = huobi_api.get_balance(self.params.symbol.name)
            if huobi_position:
                try:
                    strike = huobi_api.get_strike(f"{self.params.symbol}")
                    cutoff_price = self.params.hard_cutoff * self.gains_bag.avg_cost
                    exit_price = self.lastest_candlestick.exit_price
                    cutoff = strike <= (max(cutoff_price, exit_price))
                    if cutoff:
                        trigger_price = strike * 0.9995
                        price = trigger_price * 0.9995
                        order_type = "SL"
                        operator = "lte"
                    else:
                        max_loss = self.gains_bag.avg_cost * (
                            1 - self.params.hard_cutoff
                        )
                        min_profit = self.gains_bag.avg_cost + (
                            max_loss * self.params.profit_loss_ratio
                        )
                        if Stop_profit < min_profit:
                            print(f"{Stop_profit} -> {min_profit}")
                            Stop_profit = min_profit
                        trigger_price = None
                        price = Stop_profit * 1.0005
                        order_type = "S"
                        operator = "gte"

                    order_id = huobi_api.place_order(
                        symbol=self.params.symbol,
                        amount=huobi_position,
                        stop_price=trigger_price,
                        price=price,
                        order_type=order_type,
                        operator=operator,
                    )
                    s_sell = base_df.xBuy.notna() & base_df.xSell.isna()
                    if s_sell.any():  # should skip all False, mean nothing to update
                        base_df.loc[s_sell, "xSellOrder"] = order_id
                    else:
                        base_df.at[base_df.index[-1], "xSellOrder"] = order_id

                except Exception as e:
                    print(f"[xHunter.retreat()] place order fail: {e}")
        return base_df

    def portfolio(self, pre_strike, strike):
        return self.gains_bag.portfolio(pre_strike, strike)

    def review_mission(self, base_df):
        df = base_df[base_df.BuySignal == 1]
        sample = len(df)
        profit_sample = len(df[df.xProfit > 0])
        profit_mean = (df.xProfit > 0).mean() or ZERO
        loss_mean = (df.xProfit <= 0).mean() or ZERO
        profit_loss_ratio = profit_mean / loss_mean
        cost = self.gains_bag.init_funds
        strike = base_df.iloc[-1].Close
        profit = self.gains_bag.cash + (self.gains_bag.position * strike) - cost
        time_cost, w_daily_return = weighted_daily_return(df)
        avg_time_cost = time_cost / sample if sample else 0
        avg_profit = profit / sample if sample else 0
        drawdown = df.xProfit.min()

        _annual_return, _sortino_ratio = calc_annual_return_and_sortino_ratio(
            cost, profit, df
        )
        return pd.DataFrame(
            [
                [
                    profit,
                    cost,
                    profit_loss_ratio,
                    sample,
                    profit_sample,
                    avg_time_cost,
                    avg_profit,
                    drawdown,
                    _annual_return,
                    _sortino_ratio,
                    self.gains_bag.cash,
                    self.gains_bag.position,
                    self.gains_bag.avg_cost,
                ]
            ],
            columns=[
                "Profit",
                "Cost",
                "ProfitLossRatio",
                "Sample",
                "ProfitSample",
                "Avg.Timecost",
                "Avg.Profit",
                "Drawdown",
                "Annual.Return",
                "SortinoRatio",
                "Cash",
                "Position",
                "Avg.Cost",
            ],
        )
