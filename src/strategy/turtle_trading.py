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
    "exit_price",
    "OBV_UP",
    "VWMA",
    "Slope",
    "Slope_Diff",
    "buy",
    "sell",
    "profit",
    "time_cost",
    "Matured",
    "BuySignal",
    "P/L",
    "PRICE_VOL_reduce",
    "Rolling_Std",
    "ema_short",
    "ema_long",
    "+DM",
    "-DM",
    "ADX",
    "ADX_Signed",
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


class MixScout(IStrategyScout):
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

        # Single Bollinger band take profit solution
        surfing_df = base_df.tail(upper_sample)
        prices = surfing_df.High.values.tolist()
        mean = statistics.mean(prices)
        stdv = statistics.stdev(prices)
        surfing_profit = mean + (self.params.surfing_level * stdv)
        base_df.iloc[-1, base_df.columns.get_loc("Stop_profit")] = surfing_profit
        cut_off = base_df.Close.shift(1) - base_df.ATR.shift(1) * atr_loss_margin
        base_df.loc[idx, "exit_price"] = cut_off
        # base_df.loc[idx, "exit_price"] = np.maximum(base_df["turtle_l"], cut_off)
        return base_df

    def _calc_OBV(self, base_df, multiplier=3.4):
        window = self.params.upper_sample
        df = base_df.copy()

        # Calculate price difference and direction
        df["price_diff"] = df["Close"].diff()
        df["direction"] = df["price_diff"].apply(
            lambda x: 1 if x > 0 else -1 if x < 0 else 0
        )
        df["Vol"] *= df["direction"]

        # Calculate OBV
        df["OBV"] = df["Vol"].cumsum()

        # Calculate moving average and standard deviation for OBV
        df["OBV_MA"] = df["OBV"].rolling(window=window).mean()
        df["OBV_std"] = df["OBV"].rolling(window=window).std()

        # Calculate upper and lower bounds
        df["upper_bound"] = df["OBV_MA"] + (multiplier * df["OBV_std"])
        df["lower_bound"] = df["OBV_MA"] - (multiplier * df["OBV_std"])

        # Calculate relative difference between bounds
        df["bound_diff"] = (df["upper_bound"] - df["lower_bound"]) / df["OBV_MA"]

        df["PRICE_UP"] = df["High"] >= (
            df["Close"].rolling(window=self.params.upper_sample).mean()
            + multiplier * df["Close"].rolling(window=self.params.upper_sample).std()
        )
        df["PRICE_LOW"] = df["Low"] <= (
            df["Close"].rolling(window=self.params.lower_sample).mean()
            + multiplier * df["Close"].rolling(window=self.params.lower_sample).std()
        )

        # 计算滚动窗口内的标准差
        df["Rolling_Std"] = df["Close"].rolling(window=self.params.upper_sample).std()
        df["Rolling_Std_Percent"] = (
            df["Rolling_Std"]
            / df["Close"].rolling(window=self.params.upper_sample).mean()
        ) * 100

        # Identify significant points where OBV crosses the upper bound
        df["OBV_UP"] = (
            (df["OBV"] > df["upper_bound"])
            & (df["OBV"].shift(1) <= df["upper_bound"])
            # & (df["bound_diff"] > 0.07)
        )
        # print(df["bound_diff"])

        # Combine the new 'OBV_UP' column back to the original dataframe
        base_df["OBV"] = df["OBV"]
        # base_df["OBV_UP"] = df["OBV_UP"] & (df["Slope"] >= 0)
        # base_df["OBV_UP"] = df["OBV_UP"] & df["PRICE_UP"]
        base_df["OBV_UP"] = df["OBV_UP"] & df["PRICE_LOW"]
        # base_df["OBV_UP"] = df["OBV_UP"]
        # base_df.at[df.index[-1], "OBV_UP"] = (
        #     df["Rolling_Std_Percent"].iloc[-1]
        #     <= df["Rolling_Std_Percent"]
        #     .rolling(window=self.params.bayes_windows)
        #     .min()
        #     .iloc[-1]
        # )

        base_df["Rolling_Std"] = df["Rolling_Std_Percent"]
        base_df["upper_bound"] = df["upper_bound"]
        base_df["lower_bound"] = df["lower_bound"]
        return base_df

    def _surfing(self, base_df):
        plan_df = self.sufer.create_plan(base_df, 0)
        return plan_df

    def market_recon(self, base_df):
        base_df = pandas_util.equip_fields(base_df, TURTLE_COLUMNS)
        base_df = self._calc_ATR(base_df)
        base_df = self._calc_VWMA(base_df, window=self.params.upper_sample)
        base_df = self._calc_OBV(base_df, multiplier=self.params.atr_loss_margin)
        base_df = self._calc_profit(base_df)
        return base_df

    def _calc_VWMA(self, df, window):
        df["VWMA"] = (df.Close * df.Vol).rolling(window=window).sum() / df.Vol.rolling(
            window=window
        ).sum()
        df["Slope"] = (df.Close - df.VWMA.shift(window)) / window
        df["Slope_Diff"] = df.Slope - df.Slope.shift(1)
        return df


class TurtleScout(IStrategyScout):
    def __init__(self, params, buy_signal_func=None):
        self.params = params
        ATR_sample = self.params.ATR_sample
        upper_sample = self.params.upper_sample
        lower_sample = self.params.lower_sample
        self.window = max(ATR_sample, upper_sample, lower_sample) * 2
        # Set the buy_signal_func, default to the simple_turtle_strategy if none is provided
        self.buy_signal_func = buy_signal_func or self._simple_turtle_strategy

    def _calc_profit(self, base_df):

        surfing_df = base_df.tail(self.window).copy()
        idx = surfing_df.Stop_profit.isna().index
        prices = surfing_df.High.values.tolist()
        mean = statistics.mean(prices)
        stdv = statistics.stdev(prices)
        surfing_profit = mean + (self.params.surfing_level * stdv)
        surfing_df.loc[idx, "Stop_profit"] = surfing_profit
        surfing_df.loc[idx, "exit_price"] = (
            surfing_df.Close.shift(1)
            - surfing_df.ATR.shift(1) * self.params.atr_loss_margin
        )
        surfing_df.loc[idx, "atr_buy"] = surfing_df.Close.shift(
            1
        ) + surfing_df.ATR.shift(1)
        base_df.update(surfing_df)

        resume_idx = base_df.sell.isna().idxmax()
        df = base_df.loc[resume_idx:].copy()
        df = df[df.exit_price.notna()]

        # Use the pre-calculated BuySignal using buy_signal_func
        s_buy = df.buy.isna()
        df.loc[s_buy, "buy"] = df.Close
        df.loc[:, "BuySignal"] = self.buy_signal_func(df, self.params)

        # Sell condition:
        s_sell = df.buy.notna() & (df.Low < df.exit_price)

        df.loc[s_sell, "sell"] = df.exit_price.where(s_sell)
        df.loc[s_sell, "Matured"] = pd.to_datetime(df.Date.where(s_sell))

        # Backfill sell and Matured columns
        df.sell.bfill(inplace=True)
        df.Matured.bfill(inplace=True)

        # Compute profit and time_cost columns
        s_profit = df.buy.notna() & df.sell.notna() & df.profit.isna()
        df.loc[s_profit, "profit"] = (df.sell / df.buy) - 1
        df.loc[s_profit, "P/L"] = (df.sell - df.buy) / (
            df.ATR * self.params.atr_loss_margin
        )
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

        df = base_df.tail(self.window).copy()
        idx = df.ATR.isna().index
        df.loc[idx, "turtle_h"] = df.High.shift(1).rolling(upper_sample).max()
        df.loc[idx, "turtle_l"] = df.Low.shift(1).rolling(lower_sample).min()
        df.loc[idx, "h_l"] = df.High - df.Low
        df.loc[idx, "c_h"] = (df.Close.shift(1) - df.High).abs()
        df.loc[idx, "c_l"] = (df.Close.shift(1) - df.Low).abs()
        df.loc[idx, "TR"] = df[["h_l", "c_h", "c_l"]].max(axis=1)
        df.loc[idx, "ATR"] = df["TR"].rolling(ATR_sample).mean()
        df.loc[idx, "ATR_STDV"] = df["TR"].rolling(ATR_sample).std()
        base_df.update(df)
        return base_df

    def market_recon(self, base_df):
        base_df = pandas_util.equip_fields(base_df, TURTLE_COLUMNS)
        base_df = self._calc_ATR(base_df)
        base_df = calc_ADX(base_df, self.params, p=5)  # p=self.params.ATR_sample)
        base_df = self._calc_profit(base_df)
        return base_df

    def _simple_turtle_strategy(self, df, params):
        """
        Default simple turtle strategy to generate BuySignal.
        """
        return df.High > df.turtle_h


def emv_cross_strategy(df, params, short_windows=5, long_windws=60):
    idx = df.ema_short.isna()
    df.loc[idx, "ema_short"] = df.Close.ewm(span=short_windows, adjust=False).mean()
    df.loc[idx, "ema_long"] = df.Close.ewm(span=long_windws, adjust=False).mean()
    return (df.ema_short > df.ema_long) & (df.ADX_Signed > 0.25)


def calc_ADX(df, params, p=14):
    df = df.assign(
        **{
            "+DM": (df.High.diff().where(lambda x: x > -df.Low.diff()))
            .clip(0)
            .fillna(0),
            "-DM": (-df.Low.diff().where(lambda x: x > df.High.diff()))
            .clip(0)
            .fillna(0),
        }
    )
    df["+DM"] = df["+DM"].ewm(alpha=1 / p, adjust=False).mean()
    df["-DM"] = df["-DM"].ewm(alpha=1 / p, adjust=False).mean()
    df["+DI"] = (df["+DM"] / df.ATR).fillna(0)
    df["-DI"] = (df["-DM"] / df.ATR).fillna(0)
    df["DX"] = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])).replace(np.inf, 0)
    df["ADX"] = df.DX.ewm(alpha=1 / p, adjust=False).mean()
    df["ADX_Signed"] = df["ADX"] * np.sign(df["+DI"] - df["-DI"])
    return df
