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
        base_df.loc[idx, "atr_buy"] = base_df.Close.shift(1) + base_df.ATR.shift(1)

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
        df["OBV_DOWN"] = (
            (df["OBV"] < df["lower_bound"])
            & (df["OBV"].shift(1) >= df["lower_bound"])
            # & (df["bound_diff"] > 0.07)
        )
        # print(df["bound_diff"])

        # Combine the new 'OBV_UP' column back to the original dataframe
        base_df["PRICE_UP"] = df["PRICE_UP"]
        base_df["OBV"] = df["OBV"]
        base_df["OBV_UP"] = df["OBV_UP"]
        base_df["OBV_DOWN"] = df["OBV_DOWN"]
        # base_df["OBV_UP"] = df["OBV_UP"] & df["PRICE_UP"]
        # base_df["OBV_UP"] = df["OBV_UP"] & df["PRICE_LOW"]
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
        base_df = self._calc_OBV(base_df, multiplier=2.5)
        # base_df = self._calc_OBV(base_df, multiplier=self.params.atr_loss_margin)
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
    def __init__(self, params):
        self.params = params
        ATR_sample = self.params.ATR_sample
        upper_sample = self.params.upper_sample
        lower_sample = self.params.lower_sample
        ATR_sample = self.params.ATR_sample
        self.window = max(ATR_sample, upper_sample, lower_sample, ATR_sample) * 2

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
        atr_loss_margin = self.params.atr_loss_margin

        df = base_df.tail(self.window).copy()
        # performance: only re-calc nessasary part.
        idx = df.ATR.tail(self.window).isna().index
        df.loc[idx, "turtle_h"] = df.High.shift(1).rolling(upper_sample).max()
        df.loc[idx, "turtle_l"] = df.Low.shift(1).rolling(lower_sample).min()
        df.loc[idx, "h_l"] = df.High - df.Low
        df.loc[idx, "c_h"] = (df.Close.shift(1) - df.High).abs()
        df.loc[idx, "c_l"] = (df.Close.shift(1) - df.Low).abs()
        df.loc[idx, "TR"] = df[["h_l", "c_h", "c_l"]].max(axis=1)
        df.loc[idx, "ATR"] = df["TR"].rolling(ATR_sample).mean()
        df.loc[idx, "ATR_STDV"] = df["TR"].rolling(ATR_sample).std()  # 计算ATR的标准差

        print(df)

        # should we replace ?
        surfing_df = df.tail(upper_sample)
        prices = surfing_df.High.values.tolist()
        mean = statistics.mean(prices)
        stdv = statistics.stdev(prices)
        surfing_profit = mean + (self.params.surfing_level * stdv)
        df.loc[idx, "Stop_profit"] = surfing_profit
        # df.loc[idx, "Stop_profit"] = df.Close.shift(1) + (
        #     self.params.surfing_level * df.ATR_STDV.shift(1)
        # )
        df.loc[idx, "exit_price"] = (
            df.Close.shift(1) - df.ATR.shift(1) * atr_loss_margin
        )
        df.loc[idx, "atr_buy"] = df.Close.shift(1) + df.ATR.shift(1)
        base_df.update(df)
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
        df["OBV_DOWN"] = (
            (df["OBV"] < df["lower_bound"])
            & (df["OBV"].shift(1) >= df["lower_bound"])
            # & (df["bound_diff"] > 0.07)
        )

        # Combine the new 'OBV_UP' column back to the original dataframe
        base_df["PRICE_UP"] = df["PRICE_UP"]
        base_df["OBV"] = df["OBV"]
        base_df["OBV_UP"] = df["OBV_UP"]
        base_df["OBV_DOWN"] = df["OBV_DOWN"]

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
        base_df = self._calc_OBV(base_df, multiplier=2.5)
        # base_df = self._calc_OBV(base_df, multiplier=self.params.atr_loss_margin)
        base_df = self._calc_profit(base_df)
        return base_df


if __name__ == "__main__":

    class Params:
        ATR_sample = 5
        upper_sample = 3
        lower_sample = 3
        atr_loss_margin = 3
        surfing_level = 3

    params = Params()
    turtle_trading = TurtleScout(params)

    # Step 1: Create initial OHCLV data and test the function
    data = {
        "Open": [
            0.933081,
            0.937413,
            0.94415,
            0.947193,
            0.947025,
            0.946896,
            0.951921,
            0.953292,
            0.954867,
            0.958337,
        ],
        "High": [
            0.939477,
            0.944252,
            0.947242,
            0.947268,
            0.947462,
            0.953175,
            0.953448,
            0.955278,
            0.958683,
            0.960128,
        ],
        "Low": [
            0.932655,
            0.937335,
            0.943853,
            0.945747,
            0.945934,
            0.946807,
            0.951312,
            0.951977,
            0.954842,
            0.95807,
        ],
        "Close": [
            0.937518,
            0.944132,
            0.947242,
            0.9471,
            0.946907,
            0.95201,
            0.953316,
            0.954857,
            0.958683,
            0.960124,
        ],
        "Volume": [
            25399.5849516012,
            41409.7006304242,
            26131.1157036699,
            24556.2806726716,
            22185.9403323567,
            15747.0575832421,
            23123.9470634977,
            21584.8873799987,
            18499.1832171501,
            17063.2969134233,
        ],
        "ATR": [None] * 10,
        "Stop_profit": [None] * 10,
        "exit_price": [None] * 10,
        "atr_buy": [None] * 10,
    }
    base_df = pd.DataFrame(data)
    print(f"Before:")
    print(base_df)
    updated_df = turtle_trading._calc_ATR(base_df)
    print("Updated DataFrame after first calculation:")
    print(updated_df)

    # Step 2: Add more data and test again
    more_data = {
        "Open": [
            0.960104,
            0.95615,
            0.955749,
        ],
        "High": [
            0.960216,
            0.956406,
            0.955835,
        ],
        "Low": [
            0.955867,
            0.954181,
            0.954667,
        ],
        "Close": [
            0.956488,
            0.955758,
            0.955625,
        ],
        "Volume": [
            18616.7662870481,
            16364.0525649235,
            17956.5814010511,
        ],
        "ATR": [None] * 3,
        "Stop_profit": [None] * 3,
        "exit_price": [None] * 3,
        "atr_buy": [None] * 3,
    }
    more_df = pd.DataFrame(more_data)
    base_df = pd.concat([updated_df, more_df], ignore_index=True)
    updated_df = turtle_trading._calc_ATR(base_df)
    print("\nUpdated DataFrame after adding more data:")
    print(updated_df)

    # Step 3: Add more data and test again
    more_data = {
        "Open": [
            0.955638,
            0.954195,
            0.958453,
        ],
        "High": [
            0.955786,
            0.958146,
            0.960773,
        ],
        "Low": [
            0.953534,
            0.954179,
            0.95798,
        ],
        "Close": [
            0.954248,
            0.958146,
            0.95959,
        ],
        "Volume": [
            14470.6856807761,
            11880.1605492034,
            26549.9169391593,
        ],
        "ATR": [None] * 3,
        "Stop_profit": [None] * 3,
        "exit_price": [None] * 3,
        "atr_buy": [None] * 3,
    }
    more_df = pd.DataFrame(more_data)
    base_df = pd.concat([updated_df, more_df], ignore_index=True)
    updated_df = turtle_trading._calc_ATR(base_df)
    print("\nUpdated DataFrame after adding more data:")
    print(updated_df)
    updated_df.to_csv("test.csv")
    print(f"test.csv")
