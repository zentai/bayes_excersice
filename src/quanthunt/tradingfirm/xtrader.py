# from icecream import ic
import os
import pandas as pd
import numpy as np
import datetime
import time
import logging

from pydispatch import dispatcher
from huobi.constant.definition import OrderType, OrderState
from huobi.client.market import MarketClient
from huobi.exception.huobi_api_exception import HuobiApiException
from huobi.model.market.candlestick_event import CandlestickEvent

from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.exception.huobi_api_exception import HuobiApiException
from huobi.model.market.candlestick_event import CandlestickEvent
from ..hunterverse.interface import IHunter
from ..hunterverse.interface import xOrder, xBuyOrder, xSellOrder
from ..utils import pandas_util
from .platforms import huobi_api
from dataclasses import replace
from config import config

epsilon = 1e-8

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


def parse_client_order_id(raw_coid: str) -> tuple[str, str]:
    for suffix in ("_PROFIT_LEAVE", "_ATR_EXIT", "_CUTOFF", "_BUY"):
        if suffix in raw_coid:
            prefix = raw_coid.split("_")[0] + "_"
            return raw_coid.replace(suffix, "").replace(prefix, ""), suffix[1:]
    return raw_coid, ""


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
        self.init_position = init_position
        self.init_avg_cost = init_avg_cost
        self.position = init_position
        self.avg_cost = init_avg_cost
        self.is_sim = is_sim
        self.logger.info(
            f"Create {'Sim' if self.is_sim else 'Live'  }GainsBag: {self:snapshot}"
        )

    def reset(self):
        self.cash = self.init_funds
        self.position = self.init_position
        self.avg_cost = self.init_avg_cost

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

    def is_enough_position(self, strike):
        return (self.position * min(self.avg_cost, strike)) >= self.MIN_STAKE_CAP

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
    def __init__(self, client, params):
        self.buy_types = ("buy-stop-limit", "buy-limit", "buy-market")
        self.sell_types = ("sell-limit", "sell-stop-limit", "sell-market")
        self.status = ("filled", "partial-canceled", "partial-filled")

        self.client = client
        self.params = params
        self.dispatcher = dispatcher
        self.TOPIC_ORDER_MATCHED = "huobi_order_update"
        self.api_key = params.api_key
        self.secret_key = params.secret_key
        self.last_process_order = None
        self.benchmark_vol_for_24hour = huobi_api.get_market_detail_merged(
            self.params.symbol.name
        ).vol

        huobi_api.subscribe_order_update(
            self.params.symbol.name,
            self.api_key,
            self.secret_key,
            self.huobi_callback_orders,
            self.error,
        )

    def is_onhold(self):
        _current_vol = huobi_api.get_market_detail_merged(self.params.symbol.name).vol
        ratio = _current_vol / self.benchmark_vol_for_24hour
        if ratio < 0.6:
            print(
                f"[Volume Alert (USDT)] {_current_vol:.2f} / {self.benchmark_vol_for_24hour:.2f}, {ratio:.2f} < 0.6"
            )
            return True
        return False

    def error(self, e: "HuobiApiException"):
        print(e.error_code + e.error_message)
        print("start a new subscribe")
        huobi_api.subscribe_order_update(
            self.params.symbol.name,
            self.api_key,
            self.secret_key,
            self.huobi_callback_orders,
            self.error,
        )

    def huobi_callback_orders(self, upd_event: "OrderUpdateEvent"):
        """
        https://github.com/HuobiRDCenter/huobi_Python/blob/master/huobi/model/trade/order_update_event.py
        https://github.com/HuobiRDCenter/huobi_Python/blob/master/huobi/model/trade/order_update.py
        """

        if self.last_process_order == upd_event.data.lastActTime:
            print(
                f"remove redundant message: {upd_event.data.orderId} - {upd_event.data.clientOrderId} "
            )
            return
        else:
            self.last_process_order = upd_event.data.lastActTime

        self.dispatcher.send(
            client=self.client,
            signal=self.TOPIC_ORDER_MATCHED,
            order_id=upd_event.data.clientOrderId,
            order_status=None,
            price=None,
            position=None,
            execute_timestamp=None,
        )

    def place_order(self, order):
        def cancel(isBuy=True):
            try:
                success = (
                    huobi_api.cancel_algo_open_orders(
                        self.api_key, self.secret_key, self.params.symbol.name, isBuy
                    )
                    or []
                )
                for id in success:
                    print(f"Cancel {id} success")
            except Exception as e:
                print(f"Cancel {'Buy' if isBuy else 'Sell'} order failed: {e}")

        def place(client_id, amount, price, trigger_price, order):
            huobi_api.place_order(
                symbol=self.params.symbol,
                amount=amount,
                price=price,
                order_type=order.order_type,
                api_key=self.api_key,
                secret_key=self.secret_key,
                stop_price=trigger_price,
                client_order_id=client_id,
            )

        if isinstance(order, xBuyOrder):
            client_order_id = order.order_id
            cancel(isBuy=True)
            place(
                client_id=client_order_id,
                amount=order.position,
                price=order.target_price,
                trigger_price=order.executed_price,
                order=order,
            )

        elif isinstance(order, xSellOrder):
            client_order_id = order.order_id
            cancel(isBuy=False)
            place(
                client_id=order.order_id,
                amount=order.position,
                price=order.cutoff_price,
                trigger_price=None,
                order=order,
            )


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
        if not order:
            return
        order = replace(order)
        client = order.client

        if client not in self.order_book:
            self.order_book[client] = {"Buy": None, "Sell": None}

        if isinstance(order, xBuyOrder):
            self.order_book[client]["Buy"] = order
        elif isinstance(order, xSellOrder):
            self.order_book[client]["Sell"] = order

    def is_onhold(self):
        return False

    def match_orders(self, message):
        market_high, market_low = message.iloc[-1].High, message.iloc[-1].Low
        execute_timestamp = message.iloc[-1].Date

        for orders in self.order_book.values():
            if orders["Sell"] and orders["Sell"].status == "unfilled":
                order = orders["Sell"]
                clientOrderId = (
                    order.order_id.replace("_PROFIT_LEAVE", "")
                    .replace("_ATR_EXIT", "")
                    .replace("_CUTOFF", "")
                )

                # 按优先级检查出场价格
                if market_low <= order.cutoff_price:
                    order.status = "CUTOFF"
                    order.timestamp = execute_timestamp
                    order.executed_price = order.cutoff_price
                    self.dispatcher.send(
                        client=order.client,
                        signal=self.TOPIC_ORDER_MATCHED,
                        order_id=clientOrderId,
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
                        order_id=clientOrderId,
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
                        order_id=clientOrderId,
                        order_status=order.status,
                        price=order.profit_leave_price,
                        position=order.position,
                        execute_timestamp=order.timestamp,
                    )

            # 检查买入订单
            if orders["Buy"] and orders["Buy"].status == "unfilled":
                order = orders["Buy"]
                clientOrderId = order.order_id.replace("_BUY", "")
                if market_low <= order.target_price:
                    order.status = BUY_FILLED
                    order.executed_price = order.target_price
                    order.timestamp = execute_timestamp
                    self.dispatcher.send(
                        client=order.client,
                        signal=self.TOPIC_ORDER_MATCHED,
                        order_id=clientOrderId,
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
        self.platform = platform or SimHuobi()
        self.dispatcher = dispatcher
        self.columns = [f"{self.client}{col}" for col in HUNTER_COLUMNS]
        self.sub_market_client()
        self.watchdog = time.time()
        self.latest_candlestick = None
        self.strike = None

    def sub_market_client(self):
        print("Re-subscribing market data")
        self.market_client = MarketClient()
        self.market_client.sub_candlestick(
            self.params.symbol.name,
            pandas_util.huobi_interval.get("1min"),
            self.market_callback,
            self.market_callback_error,
        )

    def market_callback(self, candlestick_event: "CandlestickEvent"):
        try:
            self.cutoff(candlestick_event.tick.close)
            self.strike = candlestick_event.tick.close
        except Exception as e:
            print(e)

    def market_callback_error(self, e: "HuobiApiException"):
        print(e.error_code + e.error_message)

    def close_condition(self):
        position = self.gainsbag.position
        cutoff_price = self.gainsbag.cutoff_price(self.params.hard_cutoff)
        min_profit = self.gainsbag.avg_cost * (
            1 + (1 - self.params.hard_cutoff) * self.params.profit_loss_ratio
        )
        profit_leave_price = max(self.latest_candlestick.Stop_profit, min_profit)
        return (
            position,
            cutoff_price,
            profit_leave_price,
            self.latest_candlestick.exit_price,
        )

    def cutoff(self, strike):
        self.watchdog = time.time()
        if (
            not self.gainsbag.is_enough_position(strike)
            or self.latest_candlestick is None
        ):
            return False

        position, cutoff_price, profit_leave_price, atr_exit = self.close_condition()
        cid = f"{self.params.symbol.name}_{self.latest_candlestick.Date.strftime('%Y%m%d_%H%M%S')}"
        if strike <= (cutoff_price * 1.0001):
            self.load_memories(df=None)
            sell_order = xSellOrder(
                order_id=f"{cid}_CUTOFF",
                cutoff_price=min(strike, cutoff_price),
                position=position,
                order_type="S",
                client=self.client,
                atr_exit_price=0,  # not using
                profit_leave_price=0,  # not using
            )
            self.platform.place_order(sell_order)
        elif strike < (atr_exit * 1.0001):
            self.load_memories(df=None)
            sell_order = xSellOrder(
                order_id=f"{cid}_ATR_EXIT",
                cutoff_price=min(strike, atr_exit),
                position=position,
                order_type="S",
                client=self.client,
                atr_exit_price=0,  # not using
                profit_leave_price=0,  # not using
            )
            self.platform.place_order(sell_order)
        elif strike >= (profit_leave_price * 0.99):
            self.load_memories(df=None)
            sell_order = xSellOrder(
                order_id=f"{cid}_PROFIT_LEAVE",
                cutoff_price=profit_leave_price,
                position=position,
                order_type="S",
                client=self.client,
                atr_exit_price=0,  # not using
                profit_leave_price=0,  # not using
            )
            self.platform.place_order(sell_order)

    def load_memories(self, df, fetch=True):
        deals = self.params.load_deals
        start_deal = self.params.start_deal

        if not isinstance(self.platform, Huobi):
            return
        print(f"load_memories(self, fetch={fetch}, deals={deals})")

        cached_order_ids = []
        db_path = f"{config.data_dir}/{self.params.symbol}_orders.csv"
        print(f"Database path: {db_path}")

        cached_orders = pd.DataFrame()
        if os.path.exists(db_path):
            print(f"Loading cached orders from: {db_path}")
            cached_orders = pd.read_csv(db_path)

        if not cached_orders.empty:
            cached_order_ids = cached_orders.id.values.tolist()
            deals = list(set(deals) - set(cached_order_ids))

        if fetch:
            self.gainsbag.reset()
            orders = huobi_api.load_orders(
                self.params.api_key,
                self.params.secret_key,
                self.params.symbol.name,
                order_ids=deals,
            )

            # Merge cached orders with newly fetched orders
            if not cached_orders.empty:
                orders = pd.concat([cached_orders, orders], ignore_index=True)
                orders = orders.drop_duplicates(subset=["id"], keep="last")
                orders = orders.sort_values(by="finished_timestamp", ascending=True)

            start_deal_timestamp = (
                orders.loc[orders["id"] == start_deal, "finished_timestamp"].iloc[0]
                if not orders[orders["id"] == start_deal].empty
                else None
            )
            if start_deal_timestamp is not None:
                print(
                    f"Found start order {start_deal} timestamp: {start_deal_timestamp}"
                )

            for idx, order in orders.iterrows():
                if (
                    start_deal_timestamp
                    and order.finished_timestamp <= start_deal_timestamp
                ):
                    continue
                # print(f"process deals: [{idx}] - {order.client_order_id}")
                self.process_single_order(df, order)

            if not orders.empty:
                print(f"Saving {len(orders)} order records to: {db_path}")
                orders.to_csv(db_path, index=False)
                print(f"{self.gainsbag:snapshot}")

    def process_single_order(self, df, order) -> None:
        """
        【单笔订单处理】
        1. 解析订单ID并转换为 datetime，用于匹配 self.df["Date"]
        2. 根据订单类型更新 gainsbag（捕获当时快照），
        然后立即将快照回填到 df 对应行
        """
        raw_coid = getattr(order, "client_order_id", "")
        if not raw_coid:
            return

        base_coid, suffix = parse_client_order_id(raw_coid)
        try:
            order_dt = datetime.datetime.strptime(base_coid, "%Y%m%d_%H%M%S")
        except Exception as e:
            print(f"[Warning] 订单ID转换失败: {base_coid} - {e}")
            return

        p = self.client  # 用于生成字段名，例如 xBuyOrder, xSell 等

        # 处理买单：若订单ID含 "BUY" 或订单类型在 buy_types 中
        if suffix in ("BUY"):
            self.gainsbag.open_position(order.position, order.price)  # 【开仓更新】
            if df is None:
                return

            mask = df["Date"] == order_dt  # 【向量化匹配日期】
            if not mask.any():
                return

            cash_snap, avg_cost_snap, pos_snap = (
                self.gainsbag.cash,
                self.gainsbag.avg_cost,
                self.gainsbag.position,
            )
            df.loc[mask, f"{p}BuyOrder"] = raw_coid
            df.loc[mask, f"{p}Buy"] = order.price
            df.loc[mask, f"{p}Position"] = pos_snap
            df.loc[mask, f"{p}Cash"] = cash_snap
            df.loc[mask, f"{p}AvgCost"] = avg_cost_snap
            df.loc[mask, f"{p}Status"] = BUY_FILLED
            df.loc[mask, f"{p}Matured"] = order.finished_timestamp

        # 处理卖单：若订单类型在 sell_types 或订单ID含特定后缀
        elif suffix in ("PROFIT_LEAVE", "ATR_EXIT", "CUTOFF"):
            self.gainsbag.close_position(order.position, order.price)  # 【平仓更新】

            if df is None:
                return

            mask = df["Date"] == order_dt  # 【向量化匹配日期】
            if not mask.any():
                return

            cash_snap, avg_cost_snap, pos_snap = (
                self.gainsbag.cash,
                self.gainsbag.avg_cost,
                self.gainsbag.position,
            )

            df.loc[mask, f"{p}SellOrder"] = raw_coid
            df.loc[mask, f"{p}Buy"] = avg_cost_snap  # 保持买入均价
            df.loc[mask, f"{p}Sell"] = order.price
            profit = (order.price / avg_cost_snap) - 1 if avg_cost_snap else np.nan
            df.loc[mask, f"{p}Profit"] = profit
            df.loc[mask, f"{p}Position"] = pos_snap
            df.loc[mask, f"{p}Cash"] = cash_snap
            df.loc[mask, f"{p}AvgCost"] = avg_cost_snap
            df.loc[mask, f"{p}Status"] = suffix
            atr_val = df.loc[mask, "ATR"].iloc[0] if mask.any() else np.nan
            pl_value = (
                (order.price - avg_cost_snap) / (atr_val * self.params.atr_loss_margin)
                if atr_val and atr_val != 0
                else np.nan
            )
            df.loc[mask, f"{p}P/L"] = pl_value
            df.loc[mask, f"{p}Matured"] = order.finished_timestamp

    def strike_phase(self, lastest_candlestick):
        self.on_hold = self.platform.is_onhold()
        if not self.strike:
            self.strike = huobi_api.get_strike(f"{self.params.symbol}")
        self.retreat(self.strike, lastest_candlestick)
        self.attack(self.strike, lastest_candlestick)

    def attack(self, strike, lastest_candlestick):
        strike = lastest_candlestick.Kalman
        if all(
            [
                lastest_candlestick.BuySignal,
                lastest_candlestick.HMM_State == lastest_candlestick.UP_State,
                self.gainsbag.is_enough_cash(),
                not self.on_hold,
            ]
        ):
            cutoff_price = self.gainsbag.cutoff_price(self.params.hard_cutoff)
            exit_price = (
                min(lastest_candlestick.exit_price, cutoff_price)
                or lastest_candlestick.exit_price
            )
            if (strike / exit_price) < 1.002:
                print(
                    f"[Buy Rejected] strike too closes to exit_price/cutoff {strike}/{exit_price}={(strike / exit_price)}"
                )
                return

            trigger_price = strike
            kelly = 1.0  # lastest_candlestick.Kelly
            budget = self.gainsbag.discharge(kelly)
            price = trigger_price * 1.0001
            buy_order = xBuyOrder(
                order_id=f"{self.params.symbol.name}_{lastest_candlestick.Date.strftime('%Y%m%d_%H%M%S')}_BUY",
                target_price=price,
                executed_price=trigger_price,
                order_type="BL",
                operator="gte",
                kelly=kelly,
                client=self.client,
                position=budget / price,
            )
            self.platform.place_order(buy_order)

    def retreat(self, strike, latest_candlestick):
        self.latest_candlestick = latest_candlestick
        position, cutoff_price, profit_leave_price, atr_exit = self.close_condition()
        print(
            f"[{self.client}Exit]: ATR: {atr_exit}, CUT:{cutoff_price}, PROF: {profit_leave_price}, Position:{position}"
        )

        deadline = pandas_util.INTERVAL_TO_MIN.get(self.params.interval) * 60
        if (time.time() - self.watchdog) > deadline:
            print("watch dog alert")
            self.sub_market_client()

    def portfolio(self, pre_strike, strike):
        return self.gainsbag.portfolio(pre_strike, strike)

    def review_mission(self, base_df):
        hBuy = f"{self.client}Buy"
        hSell = f"{self.client}Sell"
        hProfit = f"{self.client}Profit"
        hPnL = f"{self.client}P/L"
        hStatus = f"{self.client}Status"

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
        #     f"{len(df[df.hProfit > 0])/sample:.2f}({len(df[df.hProfit > 0])}/{sample})"
        # )
        profit_mean = df[df[hProfit] >= 0][hProfit].median() or ZERO
        loss_mean = abs(df[df[hProfit] < 0][hProfit].median() or ZERO)
        strategy_profit = ((df[hPnL] > 3) & df.BuySignal).sum()
        total_profit = (df[hPnL] > 3).sum() or -1
        strategy_performance = f"{strategy_profit / total_profit:.3f}"
        # profit_loss_ratio = len(df[df.hProfit > 0]) / sample
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
                    if row[hStatus] == "ATR_EXIT" and row[hProfit] > 0
                    else (
                        "ATR_Loss"
                        if row[hStatus] == "ATR_EXIT" and row[hProfit] < 0
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
                    self.on_hold,
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
                "HOLD",
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

    # params.update(
    #     {
    #         "funds": funds,
    #         "stake_cap": cap,
    #         "symbol": Symbol(symbol),
    #         "interval": interval,
    #         "backtest": False,
    #         "debug_mode": [
    #             "statement",
    #             "statement_to_csv",
    #             "mission_review",
    #             "final_statement_to_csv",
    #         ],
    #         "load_deals": deal_ids,
    #         "api_key": "fefd13a1-bg2hyw2dfg-440b3c64-576f2",
    #         "secret_key": "1a437824-042aa429-0beff3ba-03e26",
    #     }
    # )

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
        "load_deals": [],
        "api_key": "fefd13a1-bg2hyw2dfg-440b3c64-576f2",
        "secret_key": "1a437824-042aa429-0beff3ba-03e26",
    }

    sp = StrategyParam(**params)
    x = (xHunter("x", params=sp, platform=Huobi("x", sp)),)
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
    # x = xBuyOrder(
    #     "tid",
    #     target_price=0.0000240,
    #     executed_price=0.0000240,
    #     order_type="B",
    #     operator="lte",
    #     kelly=1,
    # )

    for i in range(10):
        time.sleep(10)
        print(f"{i} --")
