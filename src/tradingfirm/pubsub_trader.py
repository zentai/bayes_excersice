# from icecream import ic
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from pydispatch import dispatcher
from huobi.constant.definition import OrderType, OrderState

from ..hunterverse.interface import IHunter
from ..utils import pandas_util
from .platforms import huobi_api

from config import config

ZERO = config.zero
BUY_FILLED = "Buy_filled"
SELL_FILLED = "Sell_filled"
CUTOFF_FILLED = "Cutoff_filled"

HUNTER_COLUMNS = [
    "sBuy",
    "sSell",
    "sProfit",
    "sPosition",
    "sCash",
    "sAvgCost",
    "sBuyOrder",
    "sSellOrder",
    "sPnLRatio",
    "sStatus",
    "xBuy",
    "xSell",
    "xProfit",
    "xPosition",
    "xCash",
    "xAvgCost",
    "xBuyOrder",
    "xSellOrder",
]


class GainsBag:
    MIN_STAKE_CAP = 10.01  # huobi limitation

    def __init__(
        self,
        symbol,
        init_funds,
        stake_cap,
        init_position=0,
        init_avg_cost=0,
        is_sim=True,
    ):
        self.logger = logging.getLogger(f"GainsBag_{symbol.name}")
        self.symbol = symbol
        self.stake_cap = stake_cap
        self.init_funds = self.cash = init_funds
        self.position = init_position
        self.avg_cost = init_avg_cost
        self.is_sim = is_sim
        self.logger.info(
            f"Create {'Sim' if self.is_sim else 'Live'  }GainsBag: {self:snapshot}"
        )

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
        # print(f"{self:snapshot}")
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

    def cutoff_price(self, cutoff_ratio=0.95):
        return self.avg_cost * cutoff_ratio

    def __format__(self, format_spec):
        if format_spec == "cash":
            return f"{self.cash}"
        elif format_spec == "avg_cost":
            return f"{self.avg_cost}"
        elif format_spec == "snapshot":
            snapshot_avg_cost = round(self.avg_cost, self.symbol.price_prec)  # USDT
            snapshot_position = round(self.position, self.symbol.amount_prec)
            snapshot_cash = round(self.cash, 2)  # USDT
            return f"![{'Sim' if self.is_sim else 'Live'  }:{self.symbol.name}] ∆ {snapshot_avg_cost} $ {snapshot_cash} Ⓒ {snapshot_position}"
        elif format_spec == "review":
            pass
        else:
            return self.name


from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SimOrder:
    order_id: str
    position: float


@dataclass
class SimBuyOrder(SimOrder):
    target_price: float
    atr_exit_price: float
    profit_leave_price: float
    order_type: str = "B"  # 默认订单类型为买入
    status: str = "unfilled"
    timestamp: datetime = field(default_factory=datetime.now)
    executed_price: Optional[float] = None  # 成交价格


@dataclass
class SimSellOrder(SimOrder):
    cutoff_price: float
    atr_exit_price: float
    profit_leave_price: float
    order_type: str = "S"  # 默认订单类型为卖出
    status: str = "unfilled"
    timestamp: datetime = field(default_factory=datetime.now)
    executed_price: Optional[float] = None  # 成交价格


class SimHuobi:
    def __init__(self):
        self.buy_order: Optional[SimBuyOrder] = None  # 当前的买入订单
        self.sell_order: Optional[SimSellOrder] = None  # 当前的卖出订单
        self.dispatcher = dispatcher
        self.dispatcher.connect(self.match_orders, signal="k_channel")

    def place_order(self, order):

        if isinstance(order, SimBuyOrder):
            # print(f"1. place order {order}")
            self.buy_order = order
            self.buy_order.timestamp = order.order_id
        elif isinstance(order, SimSellOrder):
            self.sell_order = order
            self.sell_order.timestamp = order.order_id

    def match_orders(self, message):
        market_high, market_low = message.iloc[-1].High, message.iloc[-1].Low
        execute_timestamp = message.iloc[-1].Date
        # print(f">>Received: {message.iloc[-1].Date}")
        # 检查卖出订单
        if self.sell_order and self.sell_order.status == "unfilled":
            order = self.sell_order

            # 按优先级检查出场价格
            if market_low <= order.cutoff_price:
                order.status = "CUTOFF"
                order.timestamp = execute_timestamp
                order.executed_price = order.cutoff_price
                # print(f"2. match order {order}")
                self.dispatcher.send(
                    signal="sim_order_update",
                    order_id=order.order_id,
                    order_status=order.status,
                    price=order.cutoff_price,
                    position=order.position,
                    execute_timestamp=order.timestamp,
                )
            elif market_low <= order.atr_exit_price:
                order.status = "ATR_EXIT"
                order.executed_price = order.atr_exit_price
                order.timestamp = execute_timestamp
                # print(f"2. match order {order}")
                self.dispatcher.send(
                    signal="sim_order_update",
                    order_id=order.order_id,
                    order_status=order.status,
                    price=order.atr_exit_price,
                    position=order.position,
                    execute_timestamp=order.timestamp,
                )

            elif market_high >= order.profit_leave_price:
                order.status = "Profit_LEAVE"
                order.executed_price = order.profit_leave_price
                order.timestamp = execute_timestamp
                # print(f"2. match order {order}")
                self.dispatcher.send(
                    signal="sim_order_update",
                    order_id=order.order_id,
                    order_status=order.status,
                    price=order.profit_leave_price,
                    position=order.position,
                    execute_timestamp=order.timestamp,
                )

        # 检查买入订单
        if self.buy_order and self.buy_order.status == "unfilled":
            order = self.buy_order
            if market_low <= order.atr_exit_price:
                # Maket price lower than ATR_exit, no point to buy.
                pass
            elif market_low <= self.buy_order.target_price:
                order.status = BUY_FILLED
                order.executed_price = order.target_price
                order.timestamp = execute_timestamp
                # print(f"2. match order {order}")
                self.dispatcher.send(
                    signal="sim_order_update",
                    order_id=order.order_id,
                    order_status=order.status,
                    price=order.target_price,
                    position=order.position,
                    execute_timestamp=order.timestamp,
                )


class xHunter(IHunter):
    def __init__(self, params):
        super().__init__()
        self.buy_types = ("buy-stop-limit", "buy-limit", "buy-market")
        self.sell_types = ("sell-limit", "sell-stop-limit", "sell-market")
        self.status = ("filled", "partial-canceled", "partial-filled")
        self.params = params
        self.sim_bag = GainsBag(
            symbol=params.symbol, init_funds=params.funds, stake_cap=params.stake_cap
        )
        self.live_bag = GainsBag(
            symbol=params.symbol, init_funds=params.funds, stake_cap=params.stake_cap
        )
        self.on_hold = False
        # move dispatcher connect to function call
        self.dispatcher = dispatcher
        self.dispatcher.connect(self.sim_order_update, signal="sim_order_update")
        self.dispatcher.connect(self.attack_feedback, signal="attack_feedback")

        # for sim huobi api
        self.sim_huobi = SimHuobi()

    def cutoff(self, strike):
        cutoff_price = self.live_bag.cutoff_price(self.params.hard_cutoff)
        if cutoff_price and strike <= cutoff_price:
            pass
            # TODO: actual cutoff

    def load_memories(self, fetch=True, deals=[]):
        print(f"load_memories(self, fetch={fetch}, deals={deals})")

        cached_order_ids = []
        if os.path.exists(db_path := f"{config.data_dir}/{self.params.symbol}.csv"):
            cached_order_ids = pd.read_csv(db_path).id.tolist()

        if fetch:
            orders = huobi_api.get_orders(
                set(
                    [
                        str(i)
                        for i in (
                            cached_order_ids
                            + deals
                            + huobi_api.load_history_orders(f"{self.params.symbol}")
                        )
                    ]
                )
            )

        """Process and enrich orders DataFrame with positions and prices."""
        orders = orders[orders.state.isin(self.status)]
        pd.DataFrame(orders).to_csv(db_path, index=False)

        buy_mask = orders.type.isin(self.buy_types)
        sell_mask = orders.type.isin(self.sell_types)

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

        msg = []
        for order in orders.itertuples():
            position, price, filled_cash_amount = (
                order.position,
                order.price,
                order.filled_cash_amount,
            )

            if order.type in self.buy_types:
                self.sim_bag.open_position(position, price)
                self.live_bag.open_position(position, price)
                # print(f"[B] {self.sim_bag:snapshot}")
                msg.append(["-", round(filled_cash_amount, 2), round(price, 8)])
            elif order.type in self.sell_types:
                self.sim_bag.close_position(position, price)
                self.live_bag.close_position(position, price)
                # print(f"[S] {self.sim_bag:snapshot}")
                msg.append(["+", round(filled_cash_amount, 2), round(price, 8)])

        if msg:
            fund = f"{self.sim_bag:cash}"
            cost = float(f"{self.sim_bag:avg_cost}")
            msg.append(["=", fund, cost])
            strike = huobi_api.get_strike(f"{self.params.symbol}")
            market_value = strike - cost
            msg.append(["$", market_value * position, strike])
            msg_df = pd.DataFrame(msg, columns=["@", "USDT", "Price"])
            # print(msg_df)

    def strike_phase(self, hunting_command):

        sim_retreat = False
        if "sell" in hunting_command:
            sim_retreat = self.sim_retreat(**hunting_command.get("sell"))
            # if not self.simulate:
            #     self.retreat(**hunting_command.get("sell"))
        if "buy" in hunting_command and not sim_retreat:
            self.sim_attack(**hunting_command.get("buy"))
            # if not self.simulate:
            #     self.attack(**hunting_command.get("buy"))

    def sim_order_update(
        self, order_id, order_status, price, position, execute_timestamp
    ):
        if order_status in (BUY_FILLED):
            self.sim_bag.open_position(position, price)
            # print(f"3. {order_status}: {order_id}: {self.sim_bag:snapshot}")
        # elif order_status in (SELL_FILLED, CUTOFF_FILLED):
        elif order_status in ("CUTOFF", "ATR_EXIT", "Profit_LEAVE"):
            self.sim_bag.close_position(position, price)
            # print(f"3. {order_status}: {order_id}: {self.sim_bag:snapshot}")

    # TODO: using huobi callback
    def attack_feedback(upd_event: "OrderUpdateEvent"):
        print("---- order update : ----")
        upd_event.print_object()
        print()

        # FIXME: move me to correct place
        # trade_client = TradeClient(api_key=g_api_key, secret_key=g_secret_key, init_log=True)
        # trade_client.sub_order_update("eosusdt", callback)

    def ready_move_to_upper(self, base_df):
        s_sell_order = (
            base_df.sBuyOrder.notna()
            & (base_df.sBuyOrder != "Cancel")
            & base_df.sBuy.isna()
        )

        # TODO: this should move to callback
        # expected return: order_id, position, price
        # expected behaviour: gainbag update, pub trading result + gainbag snapshot
        if s_buy_order.any():
            order_id = base_df[s_buy_order].sBuyOrder.values
            order_id = order_id[0] if order_id.any() else ""
            _order_type, _price, _position, _stop_price, _operator = order_id.split(",")
            _price = float(_price) if _price else None
            _position = float(_position) if _position else None
            if _price and (
                self.lastest_candlestick.High <= float(_price)
                or self.lastest_candlestick.Low <= float(_price)
            ):
                _price = (
                    self.lastest_candlestick.High
                    if self.lastest_candlestick.High <= float(_price)
                    else _price
                )
                print(f"[Buy Filled]: {order_id=}, {_position=}, {_price=}")
                self.sim_bag.open_position(_position, _price)
                base_df.loc[s_buy_order, "sBuy"] = _price
                base_df.loc[s_buy_order, "sPosition"] = self.sim_bag.position
                # base_df.loc[s_buy_order, "sCash"] = self.sim_bag.cash
                base_df.loc[s_buy_order, "sAvgCost"] = self.sim_bag.avg_cost
            else:
                print(f"[Buy Missed]: {order_id=}, {self.lastest_candlestick.High=}")

            # FIXME: cancel order should be somewhere
            # cancel old orders
            s_buy_order = (
                base_df.sBuyOrder.notna()
                & (base_df.sBuyOrder != "Cancel")
                & base_df.sBuy.isna()
            )
            base_df.loc[s_buy_order, "sBuyOrder"] = "Cancel"

        base_df.loc[base_df.Date == self.lastest_candlestick.Date, "sCash"] = (
            self.sim_bag.get_un_pnl(self.lastest_candlestick.Close)
        )

    def ready_to_move_turtle_scout(self):
        # TODO: is_fly should be done be turtle scout, trader only executed one price.
        """
        is_fly = price == base_df.iloc[-1].High
        price = base_df.iloc[-1].High
        position = budget / price
        if is_fly:
            order_id = f"B,{price},{position},,"
        else:
            stop_price = price * 0.9995
            order_id = f"BL,{price},{position},{stop_price},gte"
        """

    # trade_ts, target_price, kelly
    # try to book a order, create an id-orderid mapping for call back
    def sim_attack(
        self,
        hunting_id,
        target_price,
        exit_price,
        Stop_profit,
        order_type,
        kelly,
    ):
        if self.sim_bag.is_enough_cash() and not self.on_hold:
            budget = self.sim_bag.discharge(kelly)
            position = budget / target_price
            buy_order = SimBuyOrder(
                order_id=hunting_id,
                target_price=target_price,
                atr_exit_price=exit_price,
                profit_leave_price=Stop_profit,
                position=position,
                order_type=order_type,
            )
            self.sim_huobi.place_order(buy_order)

    def attack(self, base_df):
        # check Limit-buy status
        s_buy_order = (
            base_df.xBuyOrder.notna()
            & (base_df.xBuyOrder != "Cancel")
            & base_df.xBuy.isna()
        )

        # check filled order
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
                # print(f"[Buy Filled]: {order_id=}, {position=}, {price=}")
                self.live_bag.open_position(position, price)
                # print(f"live_bag: {self.live_bag:snapshot}")
                base_df.loc[s_buy_order, "xBuy"] = price
                base_df.loc[s_buy_order, "xPosition"] = self.live_bag.position
                base_df.loc[s_buy_order, "xCash"] = self.live_bag.cash
                base_df.loc[s_buy_order, "xAvgCost"] = self.live_bag.avg_cost

        # buy_signal = self.lastest_candlestick.Kelly > 0 and self.live_bag.is_enough_cash()
        buy_signal = (
            self.live_bag.is_enough_cash()
            and base_df.iloc[-1].OBV_UP == True
            and not self.on_hold
        )
        if buy_signal:
            budget = self.live_bag.discharge(ratio=1)
            # budget = self.live_bag.discharge(self.lastest_candlestick.Kelly)

            try:
                for order_type in self.buy_types:
                    success, fail = huobi_api.cancel_all_open_orders(
                        self.params.symbol.name, order_type=order_type
                    )
                    base_df.loc[base_df.xBuyOrder.isin(success), "xBuyOrder"] = "Cancel"
            except Exception as e:
                print(f"[xHunter.atack()] Cancel Old trade fail: {e}")

            try:
                # price = base_df.tail(self.params.upper_sample).High.max()
                price = base_df.tail(self.params.upper_sample).High.max()
                is_fly = price == base_df.iloc[-1].High
                price = base_df.iloc[-1].High
                position = budget / price

                if is_fly:
                    order_id = huobi_api.place_order(
                        symbol=self.params.symbol,
                        amount=position,
                        price=price,
                        order_type="B",
                    )
                else:
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

    def sim_retreat(self, hunting_id, exit_price, Stop_profit):
        if self.sim_bag.is_enough_position():
            position = self.sim_bag.position
            avg_cost = self.sim_bag.avg_cost
            min_profit = avg_cost * (
                1 + (1 - self.params.hard_cutoff) * self.params.profit_loss_ratio
            )
            final_stop_price = max(Stop_profit, min_profit)

            sell_order = SimSellOrder(
                order_id=hunting_id,
                cutoff_price=avg_cost * self.params.hard_cutoff,
                atr_exit_price=exit_price,
                profit_leave_price=final_stop_price,
                position=position,
                order_type="S",
            )
            self.sim_huobi.place_order(sell_order)

    def retreat(self, base_df):
        cutoff_price = self.params.hard_cutoff * self.live_bag.avg_cost
        cutoff_price = max(cutoff_price, self.lastest_candlestick.exit_price)
        Stop_profit = max(cutoff_price, self.lastest_candlestick.Stop_profit)

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
                _order = df_orders.loc[df_orders.id == order_id].iloc[0]
                cash = _order.filled_cash_amount
                position = _order.filled_amount
                price = cash / position
                # print(f"[Sell Order filled]: {order_id=}, {position=}, {price=}")
                self.live_bag.close_position(position, price)
                # print(f"live_bag: {self.live_bag:snapshot}")
                if _order.type == "sell-limit":
                    # print(f"mission completed: on hold")
                    self.on_hold = True
                base_df.loc[pending_order, "xSell"] = price
                base_df.loc[pending_order, "xProfit"] = (
                    base_df.xSell / base_df.xBuy
                ) - 1

        sell_signal = self.live_bag.is_enough_position()
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
                    cutoff_price = self.params.hard_cutoff * self.live_bag.avg_cost
                    exit_price = self.lastest_candlestick.exit_price
                    cutoff = strike <= (max(cutoff_price, exit_price))
                    if cutoff:
                        trigger_price = strike
                        price = trigger_price * 0.9995
                        order_type = "SL"
                        operator = "lte"
                    else:
                        max_loss = self.live_bag.avg_cost * (
                            1 - self.params.hard_cutoff
                        )
                        min_profit = self.live_bag.avg_cost + (
                            max_loss * self.params.profit_loss_ratio
                        )
                        if Stop_profit < min_profit:
                            Stop_profit = min_profit
                        trigger_price = None
                        price = Stop_profit
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
        return self.live_bag.portfolio(pre_strike, strike)

    def review_mission(self, base_df):
        def count_consecutive_losses(df):
            profits = df["sProfit"].dropna().astype(float).values
            negative = profits < 0
            neg_indices = np.where(negative)[0]

            if len(neg_indices) == 0:
                max_consecutive_losses = 0
            else:
                diff = np.diff(neg_indices)
                consecutive = np.split(neg_indices, np.where(diff != 1)[0] + 1)
                lengths = [len(c) for c in consecutive]
                max_consecutive_losses = max(lengths)

            return max_consecutive_losses

        # df = base_df[base_df.BuySignal == 1]
        df = base_df
        sample = len(df[df.sBuy.notna()]) or 0.00001
        profit_sample = "0.00"
        # profit_sample = (
        #     f"{len(df[df.sProfit > 0])/sample:.2f}({len(df[df.sProfit > 0])}/{sample})"
        # )
        profit_mean = df[df.sProfit >= 0].sProfit.median() or ZERO
        loss_mean = abs(df[df.sProfit < 0].sProfit.median() or ZERO)
        strategy_profit = ((df["P/L"] > 3) & df.BuySignal).sum()
        total_profit = (df["P/L"] > 3).sum()
        strategy_performance = f"{strategy_profit / total_profit:.3f}"
        # profit_loss_ratio = len(df[df.sProfit > 0]) / sample
        cost = self.sim_bag.init_funds
        strike = base_df.iloc[-1].Close
        profit = self.sim_bag.cash + (self.sim_bag.position * strike) - cost
        time_cost, w_daily_return = weighted_daily_return(df)
        avg_time_cost = time_cost / sample if sample else 0
        avg_profit = profit / sample if sample else 0
        drawdown = df.sProfit.min()

        _annual_return, _sortino_ratio = calc_annual_return_and_sortino_ratio(
            cost, profit, df
        )

        exit_stats = pd.DataFrame(
            {"ATR_Profit": [0], "ATR_Loss": [0], "CUTOFF": [0], "Profit_LEAVE": [0]},
            index=["Percentage"],
        ).loc["Percentage"]

        if "sStatus" in df.columns:
            df = df[df.sStatus != "Buy_filled"]
            atr_cutoff_counts = df.apply(
                lambda row: (
                    "ATR_Profit"
                    if row["sStatus"] == "ATR_EXIT" and row["sProfit"] > 0
                    else (
                        "ATR_Loss"
                        if row["sStatus"] == "ATR_EXIT" and row["sProfit"] < 0
                        else row["sStatus"]
                    )
                ),
                axis=1,
            ).value_counts()

            for category in ["ATR_Profit", "ATR_Loss", "CUTOFF", "Profit_LEAVE"]:
                if category not in atr_cutoff_counts:
                    atr_cutoff_counts[category] = 0.0
            atr_percentages = atr_cutoff_counts / atr_cutoff_counts.sum()
            exit_stats = pd.DataFrame({"Percentage": atr_percentages}).T.iloc[-1]
            profit_sample = f"{(exit_stats.Profit_LEAVE + exit_stats.ATR_Profit):.2f}"
            drawdownCount = count_consecutive_losses(df)

        return pd.DataFrame(
            [
                [
                    profit,
                    cost,
                    strategy_performance,
                    sample,
                    profit_sample,
                    avg_time_cost,
                    avg_profit,
                    drawdown,
                    drawdownCount,
                    _annual_return,
                    _sortino_ratio,
                    self.sim_bag.cash,
                    self.sim_bag.position,
                    self.sim_bag.avg_cost,
                    exit_stats.Profit_LEAVE,
                    exit_stats.ATR_Profit,
                    exit_stats.ATR_Loss,
                    exit_stats.CUTOFF,
                ]
            ],
            columns=[
                "Profit",
                "Cost",
                "StrategyPerformance",
                "Sample",
                "ProfitSample",
                "Avg.Timecost",
                "Avg.Profit",
                "Drawdown",
                "DrawdownCount",
                "Annual.Return",
                "SortinoRatio",
                "Cash",
                "Position",
                "Avg.Cost",
                "Profit_LEAVE",
                "ATR_Profit",
                "ATR_loss",
                "CUTOFF",
            ],
        )


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
