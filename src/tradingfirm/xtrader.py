# from icecream import ic
import os
import pandas as pd
import numpy as np
import logging

from pydispatch import dispatcher
from huobi.constant.definition import OrderType, OrderState

from ..hunterverse.interface import IHunter
from ..hunterverse.interface import xOrder, xBuyOrder, xSellOrder
from ..utils import pandas_util
from .platforms import huobi_api
from dataclasses import replace
from config import config

ZERO = config.zero
BUY_FILLED = "Buy_filled"
SELL_FILLED = "Sell_filled"
CUTOFF_FILLED = "Cutoff_filled"

HUNTER_COLUMNS = [
    "Buy",
    "Sell",
    "Profit",
    "Position",
    "Cash",
    "AvgCost",
    "BuyOrder",
    "SellOrder",
    "P/L",
    "Status",
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


class Huobi:
    def __init__(self, params, api_key=None, secret_key=None):
        self.buy_types = ("buy-stop-limit", "buy-limit", "buy-market")
        self.sell_types = ("sell-limit", "sell-stop-limit", "sell-market")
        self.status = ("filled", "partial-canceled", "partial-filled")

        self.params = params
        self.dispatcher = dispatcher
        self.TOPIC_ORDER_MATCHED = "sim_order_update"
        self.api_key = api_key or "fefd13a1-bg2hyw2dfg-440b3c64-576f2"
        self.secret_key = secret_key or "1a437824-042aa429-0beff3ba-03e26"

        self.order_book = {
            # client: {
            #     Buy: xBuyOrder,
            #     Sell: xSellOrder,
            # }
        }

        huobi_api.subscribe_order_update(
            self.params.symbol.name,
            self.api_key,
            self.secret_key,
            self.huobi_callback_orders,
        )

    def huobi_callback_orders(self, upd_event: "OrderUpdateEvent"):
        """
        The detail order information.

        :member
            orderId: The order id.
            tradePrice: trade price
            tradeVolume: trade volume
            tradeId: Id record for trade
            tradeTime: trade timestamp (ms)
            aggressor: true (taker), false (maker)
            remainAmt: Remaining amount (for buy-market order it's remaining value)
            orderStatus: Order status, valid value: partial-filled, filled
            clientOrderId: Client order ID (if any)
            eventType: Event type, valid value: trade
            symbol: The symbol, like "btcusdt".
            type: The order type, possible values are: buy-market, sell-market, buy-limit, sell-limit, buy-ioc, sell-ioc, buy-limit-maker, sell-limit-maker, buy-limit-fok, sell-limit-fok.

            {'accountId': 0,
            'aggressor': False,
            'clientOrderId': '',
            'eventType': 'trade',
            'orderId': 1226305230997968,
            'orderStatus': 'filled',
            'remainAmt': '0',
            'symbol': 'pepeusdt',
            'tradeId': 43798743,
            'tradePrice': '0.000024',
            'tradeTime': 1734387557456,
            'tradeVolume': '520833.33',
            'type': 'buy-limit'}
        """
        if upd_event.data.orderStatus in ("partial-filled", "filled"):
            print(
                {
                    "orderId": upd_event.data.orderId,
                    "tradePrice": upd_event.data.tradePrice,
                    "tradeVolume": upd_event.data.tradeVolume,
                    "tradeId": upd_event.data.tradeId,
                    "tradeTime": upd_event.data.tradeTime,
                    "aggressor": upd_event.data.aggressor,
                    "remainAmt": upd_event.data.remainAmt,
                    "orderStatus": upd_event.data.orderStatus,
                    "clientOrderId": upd_event.data.clientOrderId,
                    "eventType": upd_event.data.eventType,
                    "symbol": upd_event.data.symbol,
                    "type": upd_event.data.type,
                    "accountId": upd_event.data.accountId,
                }
            )
            _order_status = None
            if "PROFIT_LEAVE" in upd_event.data.clientOrderId:
                _order_status = "Profit_LEAVE"
            elif "ATR_EXIT" in upd_event.data.clientOrderId:
                _order_status = "ATR_EXIT"
            elif "CUTOFF" in upd_event.data.clientOrderId:
                _order_status = "CUTOFF"

            print(f"{_order_status}")

            self.dispatcher.send(
                client=order.client,
                signal=self.TOPIC_ORDER_MATCHED,
                order_id=upd_event.data.orderId,
                order_status=_order_status,
                price=order.atr_exit_price,
                position=order.position,
                execute_timestamp=order.timestamp,
            )

    def place_order(self, order):
        order = replace(order)  # clone
        client = order.client

        if client not in self.order_book:
            self.order_book[client] = {"Buy": None, "Sell": None}

        if isinstance(order, xBuyOrder):
            try:
                cancel_order = self.order_book[client]["Buy"]
                success, fail = huobi_api.cancel_algo_open_orders(
                    self.api_key, self.secret_key, [cancel_order.order_id]
                )
                print(f"cancel buy orders: {success}, fail: {fail}")
                # for order_type in self.buy_types:
                #     success, fail = huobi_api.cancel_all_open_orders(
                #         self.params.symbol.name,
                #         self.api_key,
                #         self.secret_key,
                #         order_type=order_type,
                #     )
            except Exception as e:
                print(f"Cancel buy order fail: {e}")

            order_id = huobi_api.place_order(
                symbol=self.params.symbol,
                amount=order.position,
                price=order.target_price,
                order_type=order.order_type,
                api_key=self.api_key,
                secret_key=self.secret_key,
                stop_price=order.executed_price,
                client_order_id=f"{order.order_id}_BUY",
            )
            print(f"HTX buy order id: {order_id}")
            self.order_book[client]["Buy"] = order

        elif isinstance(order, xSellOrder):
            try:
                cancel_order = self.order_book[client]["Sell"]
                success, fail = huobi_api.cancel_algo_open_orders(
                    self.api_key,
                    self.secret_key,
                    [
                        f"{cancel_order.order_id}_ATR_EXIT",
                        f"{cancel_order.order_id}_PROFIT_LEAVE",
                    ],
                )
                print(f"cancel sell orders: {success}, fail: {fail}")
                # for order_type in self.sell_types:
                #     success, fail = huobi_api.cancel_all_open_orders(
                #         self.params.symbol.name,
                #         self.api_key,
                #         self.secret_key,
                #         order_type=order_type,
                #     )
                #     # TODO： base_df.loc[base_df.xBuyOrder.isin(success), "xBuyOrder"] = "Cancel"
            except Exception as e:
                print(f"Cancel sell order fail: {e}")

            # place ATR exit order
            order_id = huobi_api.place_order(
                symbol=self.params.symbol,
                amount=order.position,
                stop_price=order.atr_exit_price * 0.0095,
                price=order.atr_exit_price,
                order_type=order.order_type,
                operator="lte",
                client_order_id=f"{order.order_id}_ATR_EXIT",
            )
            # place profit exit order
            order_id = huobi_api.place_order(
                symbol=self.params.symbol,
                amount=order.position,
                stop_price=order.profit_leave_price * 0.0095,
                price=order.profit_leave_price,
                order_type=order.order_type,
                operator="gte",
                client_order_id=f"{order.order_id}_PROFIT_LEAVE",
            )
            # place cutoff exit order
            order_id = huobi_api.place_order(
                symbol=self.params.symbol,
                amount=order.position,
                stop_price=order.cutoff_price * 0.0095,
                price=order.cutoff_price,
                order_type=order.order_type,
                operator=order.operator,
                client_order_id=f"{order.order_id}_CUTOFF",
            )
            self.order_book[client]["Sell"] = order


class SimHuobi:
    def __init__(self):
        self.dispatcher = dispatcher
        self.dispatcher.connect(self.match_orders, signal="k_channel")
        self.TOPIC_ORDER_MATCHED = "sim_order_update"
        self.order_book = {
            # client: {
            #     Buy: xBuyOrder,
            #     Sell: xSellOrder,
            # }
        }

    def place_order(self, order):
        order = replace(order)
        client = order.client

        if client not in self.order_book:
            self.order_book[client] = {"Buy": None, "Sell": None}

        if isinstance(order, xBuyOrder):
            self.order_book[client]["Buy"] = order
        elif isinstance(order, xSellOrder):
            self.order_book[client]["Sell"] = order

    def match_orders(self, message):
        market_high, market_low = message.iloc[-1].High, message.iloc[-1].Low
        execute_timestamp = message.iloc[-1].Date

        for orders in self.order_book.values():
            if orders["Sell"] and orders["Sell"].status == "unfilled":
                order = orders["Sell"]

                # 按优先级检查出场价格
                if market_low <= order.cutoff_price:
                    order.status = "CUTOFF"
                    order.timestamp = execute_timestamp
                    order.executed_price = order.cutoff_price
                    self.dispatcher.send(
                        client=order.client,
                        signal=self.TOPIC_ORDER_MATCHED,
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
                    self.dispatcher.send(
                        client=order.client,
                        signal=self.TOPIC_ORDER_MATCHED,
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
                    self.dispatcher.send(
                        client=order.client,
                        signal=self.TOPIC_ORDER_MATCHED,
                        order_id=order.order_id,
                        order_status=order.status,
                        price=order.profit_leave_price,
                        position=order.position,
                        execute_timestamp=order.timestamp,
                    )

            # 检查买入订单
            if orders["Buy"] and orders["Buy"].status == "unfilled":
                order = orders["Buy"]
                if market_low <= order.atr_exit_price:
                    # Maket price lower than ATR_exit, no point to buy.
                    pass
                elif market_low <= order.target_price:
                    order.status = BUY_FILLED
                    order.executed_price = order.target_price
                    order.timestamp = execute_timestamp
                    self.dispatcher.send(
                        client=order.client,
                        signal=self.TOPIC_ORDER_MATCHED,
                        order_id=order.order_id,
                        order_status=order.status,
                        price=order.target_price,
                        position=order.position,
                        execute_timestamp=order.timestamp,
                    )


class xHunter(IHunter):
    def __init__(self, client, params, platform=None):
        super().__init__()
        self.client = client
        self.buy_types = ("buy-stop-limit", "buy-limit", "buy-market")
        self.sell_types = ("sell-limit", "sell-stop-limit", "sell-market")
        self.status = ("filled", "partial-canceled", "partial-filled")
        self.params = params
        self.gainsbag = GainsBag(
            symbol=params.symbol, init_funds=params.funds, stake_cap=params.stake_cap
        )
        self.on_hold = False
        # move dispatcher connect to function call
        self.platform = platform or SimHuobi()
        self.dispatcher = dispatcher
        self.dispatcher.connect(
            self.callback_order_matched, signal=self.platform.TOPIC_ORDER_MATCHED
        )
        self.columns = [f"{self.client}{col}" for col in HUNTER_COLUMNS]
        if self.params.load_deals:
            self.load_memories(deals=self.params.load_deals)

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
        retreat = False
        if "sell" in hunting_command:
            retreat = self.retreat(hunting_command.get("sell"))
        if "buy" in hunting_command and not retreat:
            self.attack(hunting_command.get("buy"))

    def callback_order_matched(
        self, client, order_id, order_status, price, position, execute_timestamp
    ):
        if client != self.client:
            return
        if order_status in (BUY_FILLED):
            self.gainsbag.open_position(position, price)
        elif order_status in ("CUTOFF", "ATR_EXIT", "Profit_LEAVE"):
            self.gainsbag.close_position(position, price)

    # trade_ts, target_price, kelly
    # try to book a order, create an id-orderid mapping for call back
    def attack(self, buy_order):
        if self.gainsbag.is_enough_cash() and not self.on_hold:
            budget = self.gainsbag.discharge(buy_order.kelly)
            position = budget / buy_order.target_price
            buy_order.client = self.client
            buy_order.position = position
            self.platform.place_order(buy_order)

    def retreat(self, sell_order):
        if self.gainsbag.is_enough_position():
            position = self.gainsbag.position
            avg_cost = self.gainsbag.avg_cost
            min_profit = avg_cost * (
                1 + (1 - self.params.hard_cutoff) * self.params.profit_loss_ratio
            )

            sell_order.client = self.client
            sell_order.profit_leave_price = max(
                sell_order.profit_leave_price, min_profit
            )
            sell_order.cutoff_price = avg_cost * self.params.hard_cutoff
            sell_order.position = position
            self.platform.place_order(sell_order)

    def portfolio(self, pre_strike, strike):
        return self.gainsbag.portfolio(pre_strike, strike)

    def review_mission(
        self,
        base_df,
        hBuy="sBuy",
        hSell="sSell",
        hProfit="sProfit",
        hPnL="sP/L",
        hStatus="sStatus",
    ):
        def count_consecutive_losses(df):
            profits = df[hProfit].dropna().astype(float).values
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
        sample = df[hBuy].notna().sum() or 0.00001
        profit_sample = "0.00"
        # profit_sample = (
        #     f"{len(df[df.sProfit > 0])/sample:.2f}({len(df[df.sProfit > 0])}/{sample})"
        # )
        profit_mean = df[df[hProfit] >= 0][hProfit].median() or ZERO
        loss_mean = abs(df[df[hProfit] < 0][hProfit].median() or ZERO)
        strategy_profit = ((df[hPnL] > 3) & df.BuySignal).sum()
        total_profit = (df[hPnL] > 3).sum()
        strategy_performance = f"{strategy_profit / total_profit:.3f}"
        # profit_loss_ratio = len(df[df.sProfit > 0]) / sample
        cost = self.gainsbag.init_funds
        strike = base_df.iloc[-1].Close
        profit = self.gainsbag.cash + (self.gainsbag.position * strike) - cost
        time_cost, w_daily_return = weighted_daily_return(df)
        avg_time_cost = time_cost / sample if sample else 0
        avg_profit = profit / sample if sample else 0
        drawdown = df[hProfit].min()

        _annual_return, _sortino_ratio = calc_annual_return_and_sortino_ratio(
            cost, profit, df
        )

        exit_stats = pd.DataFrame(
            {"ATR_Profit": [0], "ATR_Loss": [0], "CUTOFF": [0], "Profit_LEAVE": [0]},
            index=["Percentage"],
        ).loc["Percentage"]

        if hStatus in df.columns:
            df = df[df[hStatus] != "Buy_filled"]
            atr_cutoff_counts = df.apply(
                lambda row: (
                    "ATR_Profit"
                    if row[hStatus] == "ATR_EXIT" and row["sProfit"] > 0
                    else (
                        "ATR_Loss"
                        if row[hStatus] == "ATR_EXIT" and row["sProfit"] < 0
                        else row[hStatus]
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
                    self.gainsbag.cash,
                    self.gainsbag.position,
                    self.gainsbag.avg_cost,
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
        (1 + df_subset["sProfit"]) ** (1 / df_subset["time_cost"])
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
        (df.Kelly > 0) & (df.sProfit < _yield_curve_1yr)
    ].sProfit.std(ddof=1)
    _annual_downside_risk_stdv = _downside_risk_stdv * np.sqrt(_annual_trade_count)
    t = _trade_minutes / (365 * 24 * 60)
    _annual_return = (profit / cost) ** (1 / t) - 1 if (profit / cost > 0) else 0
    _sortino_ratio = (_annual_return - _yield_curve_1yr) / _annual_downside_risk_stdv
    return _annual_return, _sortino_ratio


if __name__ == "__main__":
    from hunterverse.interface import Symbol
    from hunterverse.interface import StrategyParam
    import time

    params = {
        # Buy
        "ATR_sample": 12,
        "bayes_windows": 12,
        "lower_sample": 12,
        "upper_sample": 12,
        # Sell
        "hard_cutoff": 0.9,
        "profit_loss_ratio": 3,
        "atr_loss_margin": 2,
        "surfing_level": 7,
        # Period
        "interval": "1min",
        "funds": 100,
        "stake_cap": 12.50,
        "symbol": Symbol("pepeusdt"),
        "backtest": False,
        "debug_mode": [
            "statement",
            "statement_to_csv",
            "mission_review",
            "final_statement_to_csv",
        ],
    }

    # sp = StrategyParam(**params)
    # x = xHunter("x", sp, Huobi(sp))
    # x.attack(
    #     xBuyOrder(
    #         "tid",
    #         target_price=0.0000240,
    #         executed_price=0.0000240,
    #         order_type="B",
    #         operator="lte",
    #         kelly=1,
    #     )
    # )
    # for i in range(2000):
    #     time.sleep(10)
    x = xBuyOrder(
        "tid",
        target_price=0.0000240,
        executed_price=0.0000240,
        order_type="B",
        operator="lte",
        kelly=1,
    )

    y = replace(x)
    y.order_id = "xdrser"
    print(x)
    print(y)
