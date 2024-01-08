from story import IHunter, DEBUG_COL
from utils import pandas_util
import pandas as pd
import logging

HUNTER_COLUMNS = [
    "xBuy",
    "xSell",
    "xProfit",
    "xPosition",
    "xCash",
    "xAvgCost",
]


class GainsBag:
    MIN_STAKE_CAP = 10  # huobi limitation

    def __init__(self, symbol, init_funds, stake_cap, init_position=0, init_avg_cost=0):
        self.logger = logging.getLogger(f"GainsBag_{symbol}")
        self.symbol = symbol
        self.stake_cap = stake_cap
        self.init_funds = self.cash = init_funds
        self.position = init_position
        self.avg_cost = init_avg_cost
        self.logger.info(
            f"Create GainsBag: [{self.symbol}] ${self.init_funds} Ⓒ {self.position} ∆ {self.avg_cost}"
        )

    def get_un_pnl(self, strike):
        return (strike * self.position) + self.cash - self.init_funds

    def log_action(self, action, price, position, cash):
        price = round(price, self.symbol.price_prec)
        position = round(position, self.symbol.amount_prec)
        cash = round(cash, 2)  # USDT
        unpnl = round(self.get_un_pnl(price), 2)  # USDT
        msg = f"[{self.symbol}] ∆ {price} $ {cash} Ⓒ {position} unPNL: {unpnl}"
        if action == "open_position":
            msg = f"+{msg}"
        elif action == "close_position":
            msg = f"-{msg}"
        self.logger.info(msg)

        # dump snapshot
        snapshot_avg_cost = round(self.avg_cost, 2)  # USDT
        snapshot_position = round(self.position, self.symbol.amount_prec)
        snapshot_cash = round(self.cash, 2)  # USDT
        self.logger.info(
            f"![{self.symbol}] ∆ {snapshot_avg_cost} $ {snapshot_cash} Ⓒ {snapshot_position}"
        )

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


class xHunter(IHunter):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.fetch_huobi = params.fetch_huobi
        self.simulate = params.simulate
        self.gains_bag = GainsBag(symbol=params.symbol, init_funds=200, stake_cap=50)

    def strike_phase(self, base_df):
        base_df = pandas_util.equip_fields(base_df, HUNTER_COLUMNS)
        lastest_candlestick = base_df.iloc[-1]
        buy_signal = lastest_candlestick.Kelly > 0
        sell_signal = lastest_candlestick.Low < lastest_candlestick.exit_price
        if buy_signal:
            budget = self.gains_bag.discharge(lastest_candlestick.Kelly)
            if budget:
                if self.params.simulate:
                    price = (
                        pandas_util.sim_trade(self.params.symbol, action="buy")
                        if self.params.fetch_huobi
                        else lastest_candlestick.Close
                    )
                    position = budget / price
                    self.gains_bag.open_position(position, price)
                    s_buy = base_df.Date == lastest_candlestick.Date
                    base_df.loc[s_buy, "xBuy"] = price
                    base_df.loc[s_buy, "xPosition"] = self.gains_bag.position
                    base_df.loc[s_buy, "xCash"] = self.gains_bag.cash
                    base_df.loc[s_buy, "xAvgCost"] = self.gains_bag.avg_cost
                else:
                    # TODO: call huobi client, using market price to buy
                    # base_df.loc[base_df.Date==lastest_candlestick.Date, 'xBuy'] = pandas_util.sim_trade(self.params.symbol, action='buy')
                    pass
        elif sell_signal:
            position = self.gains_bag.position
            if position:
                if self.params.simulate:
                    price = (
                        pandas_util.sim_trade(self.params.symbol, action="sell")
                        if self.params.fetch_huobi
                        else lastest_candlestick.exit_price
                    )
                    self.gains_bag.close_position(position, price)
                    s_sell = base_df.xBuy.notna() & base_df.xSell.isna()
                    last_index = base_df.loc[s_sell].index[-1]
                    base_df.loc[s_sell, "xSell"] = price
                    base_df.loc[s_sell, "xProfit"] = (base_df.xSell / base_df.xBuy) - 1
                    base_df.at[last_index, "xPosition"] = self.gains_bag.position
                    base_df.at[last_index, "xCash"] = self.gains_bag.cash
                else:
                    # TODO: call huobi client, using market price to sell
                    # base_df.loc[base_df.Date==lastest_candlestick.Date, 'xBuy'] = pandas_util.sim_trade(self.params.symbol, action='buy')
                    pass
        return base_df
