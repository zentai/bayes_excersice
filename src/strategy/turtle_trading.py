import sys
import os
import numpy as np
import pandas as pd
import statistics
from datetime import datetime
from collections import deque
import logging

from ..hunterverse.interface import IStrategyScout
from ..utils import pandas_util

logger = logging.getLogger(__name__)


# FIXME: move some columns to IEngine
TURTLE_COLUMNS = [
    "ATR",
    "turtle_h",
    "turtle_l",
    "Stop_profit",
    "surfing_buy",
    "surfing_sell",
    "buy",
    "sell",
    "profit",
    "time_cost",
    "Matured",
    "BuySignal",
]


class BollingerBandsSurfer:
    def __init__(self, symbol, buy_sell_conf, buffer_size=30):
        self.symbol = symbol
        self.buy_sell_conf = buy_sell_conf
        self.stdv_ring_buffer = deque(maxlen=buffer_size)

    def create_plan(self, prices_df, min_profit_price):
        prices = prices_df.Close.values.tolist()
        mean = statistics.mean(prices)
        stdv = statistics.stdev(prices)
        self.stdv_ring_buffer.append(stdv)
        if stdv < statistics.median(self.stdv_ring_buffer):
            logger.info(
                f"Stdv: {stdv} < Median Stdv: {statistics.median(self.stdv_ring_buffer)}, stop buy sell plan"
            )
            return pd.DataFrame()
        bs_conf = self.buy_sell_conf.copy()
        bs_conf.loc[bs_conf.Action == "S", "price"] = min_profit_price + (
            bs_conf["level"] * stdv
        )
        bs_conf.loc[bs_conf.Action == "B", "price"] = mean + (bs_conf["level"] * stdv)
        return bs_conf


class TurtleScout(IStrategyScout):
    def __init__(self, params):
        self.params = params
        ma_window = 10
        trade_conf = [
            ["S", 10, 0.25],
            ["S", 7, 0.25],
            ["S", 5, 0.5],
            ["B", -5, 0.125],
            ["B", -7, 0.125],
            ["B", -9, 0.25],
            ["B", -11, 0.5],
        ]
        trade_conf = pd.DataFrame(trade_conf, columns=["Action", "level", "ratio"])
        self.sufer = BollingerBandsSurfer(params.symbol.name, trade_conf, ma_window)

    def _calc_profit(self, base_df):
        resume_idx = base_df.sell.isna().idxmax()
        df = base_df.loc[resume_idx:].copy()

        # Buy daily basic when the market price is higher than Stop profit
        s_buy = (
            # df.buy.isna() & df.exit_price.notna() & (df.exit_price < df.Open.shift(-1))
            # df.buy.isna() & df.exit_price.notna() & (df.exit_price < df.Close)
            df.buy.isna()
            & df.exit_price.notna()
        )
        # df.loc[s_buy, "buy"] = df.Open.shift(-1)
        df.loc[s_buy, "buy"] = df.Close
        df.loc[:, "BuySignal"] = df.High > df.turtle_h
        # Sell condition:
        # s_sell = df.buy.notna() & (df.Low.shift(-1) < df.exit_price)
        s_sell = df.buy.notna() & (df.Close < df.exit_price)
        df.loc[s_sell, "sell"] = df.exit_price.where(s_sell)
        # df.loc[s_sell, "Matured"] = pd.to_datetime(df.Date.shift(-1).where(s_sell))
        df.loc[s_sell, "Matured"] = pd.to_datetime(df.Date.where(s_sell))

        # Backfill sell and Matured columns
        df.sell.bfill(inplace=True)
        df.Matured.bfill(inplace=True)

        # Compute profit and time_cost columns
        s_profit = df.buy.notna() & df.sell.notna() & df.profit.isna()
        df.loc[s_profit, "profit"] = (df.sell / df.buy) - 1
        df.loc[s_profit, "time_cost"] = [
            int(x.seconds / 60 / pandas_util.INTERVAL_TO_MIN.get(self.params.interval))
            for x in (
                pd.to_datetime(df.loc[s_profit, "Matured"])
                - pd.to_datetime(df.loc[s_profit, "Date"])
            )
        ]

        # Clear sell and Matured values where buy is NaN
        df.loc[df.buy.isna(), "sell"] = np.nan
        df.loc[df.buy.isna(), "Matured"] = pd.NaT
        base_df.update(df)
        return base_df

    def _calc_ATR(self, base_df):
        ATR_sample = self.params.ATR_sample
        upper_sample = self.params.upper_sample
        lower_sample = self.params.lower_sample
        ATR_sample = self.params.ATR_sample
        atr_loss_margin = self.params.atr_loss_margin

        # performance: only re-calc nessasary part.
        idx = (
            base_df.index
            if base_df.ATR.isna().all()
            else base_df.ATR.iloc[ATR_sample:].isna().index
        )
        base_df.loc[idx, "turtle_h"] = base_df.High.shift(1).rolling(upper_sample).max()
        base_df.loc[idx, "turtle_l"] = base_df.Low.shift(1).rolling(lower_sample).min()
        base_df.loc[idx, "h_l"] = base_df.High - base_df.Low
        base_df.loc[idx, "c_h"] = (base_df.Close.shift(1) - base_df.High).abs()
        base_df.loc[idx, "c_l"] = (base_df.Close.shift(1) - base_df.Low).abs()
        base_df.loc[idx, "TR"] = base_df[["h_l", "c_h", "c_l"]].max(axis=1)
        base_df.loc[idx, "ATR"] = base_df["TR"].rolling(ATR_sample).mean()
        base_df.loc[idx, "Stop_profit"] = (
            base_df.Close.shift(1) - base_df.ATR.shift(1) * atr_loss_margin
        )
        base_df.loc[idx, "exit_price"] = base_df[["turtle_l", "Stop_profit"]].max(
            axis=1
        )
        return base_df

    def _surfing(self, base_df):
        plan_df = self.sufer.create_plan(base_df, 0)
        return plan_df

    def market_recon(self, base_df):
        base_df = pandas_util.equip_fields(base_df, TURTLE_COLUMNS)
        base_df['surfing_buy'] = base_df['surfing_buy'].astype(object)
        base_df['surfing_sell'] = base_df['surfing_sell'].astype(object)
        base_df = self._calc_ATR(base_df)
        base_df = self._calc_profit(base_df)
        # TODO
        # migrate boll buy/sell plan into base_df
        # base_df = pandas_util.equip_fields(base_df, BOLL_SURF_COLUNBS)
        records = self._surfing(base_df).to_dict('records')
        sell_params = [ (i['ratio'], i['price']) for i in records if i['Action'] == 'S']
        buy_params = [ (i['ratio'], i['price']) for i in records if i['Action'] == 'B']
        _index = len(base_df) - 1
        base_df.at[_index, 'surfing_buy'] = buy_params
        base_df.at[_index, 'surfing_sell'] = sell_params
        print(base_df)
        # base_df = self._calc_surf_profit(base_df)   #TODO: this is a good solution?
        
        return base_df
