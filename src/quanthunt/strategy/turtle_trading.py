import sys
import os
import numpy as np
import pandas as pd
import statistics
from datetime import datetime
from collections import deque
import logging
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from quanthunt.hunterverse.interface import IStrategyScout
from quanthunt.utils import pandas_util

logger = logging.getLogger(__name__)
epsilon = 1e-8

# FIXME: move some columns to IEngine
TURTLE_COLUMNS = [
    "TR",
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
    "drift",
    "volatility",
    "pred_price",
    "Kalman",
    "log_returns",
    "global_log_volatility",
    "global_log_vol",
    "KReturn",
    "KReturnVol",
    "RVolume",
    "HMM_State",
    "UP_State",
    "adjusted_margin",
    "Count_Hz",
    "Amount_Hz",
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
        df.loc[s_buy, "BuySignal"] = df.High > df.turtle_h
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
        surf_df = base_df.tail(upper_sample)
        prices = surf_df.High.values.tolist()
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
        self.scaler = None
        self.hmm_model = None

    def train(self, df):
        windows = self.params.bayes_windows
        df = pandas_util.equip_fields(df, TURTLE_COLUMNS)
        df = self._calc_ATR(df)
        df = self._calc_OBV(df)

        self.scaler = StandardScaler()
        self.hmm_model = GaussianHMM(
            n_components=2, covariance_type="full", n_iter=4000, random_state=42
        )

        X = self._calc_HMM_input(df, windows, self.scaler)
        self.hmm_model.fit(X)
        hidden_states = self.hmm_model.predict(X)
        df["HMM_State"] = hidden_states

        # debug_hmm(self.hmm_model, hidden_states, df, X)

        HMM_0 = df[df.HMM_State == 0].Slope.median()
        HMM_1 = df[df.HMM_State == 1].Slope.median()
        df["UP_State"] = int(HMM_1 > HMM_0)
        df = self._calc_profit(df)
        return df

    def _calc_HMM_input(self, df, windows, scaler):
        df = calc_Kalman_Price(df, windows=windows)
        mask = df["KReturn"].isna()
        df.loc[mask, "KReturn"] = np.log(
            # df.loc[mask, "Kalman"] / df["Kalman"].shift(1)[mask]
            df.loc[mask, "Kalman"]
            / (
                df["Kalman"].shift(1).rolling(window=60, min_periods=5).mean()[mask]
                + epsilon
            )
        )

        df.loc[mask, "KReturnVol"] = (
            df["KReturn"].rolling(window=windows, min_periods=5).mean()
        )

        # 计算 RVolume：log(Vol / Vol_rolling), Vol_rolling 为 Vol 的 rolling 均值
        # vol_rolling = df.loc[mask, "Vol"].rolling(window=windows, min_periods=1).mean()
        # df.loc[mask, "RVolume"] = df.loc[mask, "Vol"] / vol_rolling

        rolling_mean = (
            df.loc[mask, "Vol"].rolling(window=windows, min_periods=5).mean() + epsilon
        )
        df.loc[mask, "RVolume"] = np.log((df.loc[mask, "Vol"] + epsilon) / rolling_mean)

        # df.loc[mask, "RVolume"] = df["volatility"][mask]

        # 填充 KReturnVol 和 RVolume 的缺失值, 采用向后填充策略
        df.loc[:, ["KReturnVol", "RVolume", "Slope", "OBV_MA"]] = df.loc[
            :, ["KReturnVol", "RVolume", "Slope", "OBV_MA"]
        ].fillna(method="bfill")

        # features = df[["KReturnVol", "RVolume", "Slope"]].values
        features = df[["KReturnVol", "RVolume", "Slope", "OBV_MA"]].values
        X = scaler.fit_transform(features)
        return X

    def adjust_exit_price_by_slope(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【动态止损】【趋势强弱】【Kalman】【Slope】【HMM】
        根据趋势强弱动态调整 ATR 平仓门槛，并计算 exit_price。

        - 使用已存在的 Slope 指标（来自 Kalman 与 local_low）
        - 趋势强（HMM_State == UP_State）：使用最大 ATR 倍数
        - 趋势弱：在 atr_margin_min ~ atr_margin_max 间插值
        - 自动 clip Slope 值，避免极端值失控
        - 输出更新后的 df，包含 adjusted_margin 与 exit_price
        """

        # ===【参数载入】===
        up_state = df.iloc[-1].UP_State  # 关键字：HMM、上升趋势状态
        slope_col = "Slope"
        hmm_col = "HMM_State"
        atr_margin_max = self.params.atr_loss_margin  # 关键字：ATR最大门槛
        atr_margin_min = getattr(
            self.params, "atr_margin_min", 0.1
        )  # 可调，关键字：最小止损门槛

        # ===【趋势强弱】Slope 已预先存在，避免重复计算===
        slope = df[slope_col]

        # ===【clip：限制斜率范围，避免异常波动】===
        slope_clip_min = slope.quantile(0.05)  # 趋势最弱
        slope_clip_max = slope.quantile(0.50)  # 平稳趋势或微升

        slope_clipped = slope.clip(slope_clip_min, slope_clip_max)

        # ===【normalized：映射到0~1之间，用来插值】===
        normalized = (slope_clipped - slope_clip_min) / (
            slope_clip_max - slope_clip_min
        )

        # ===【动态止损门槛：趋势越弱 → margin 越小 → 越容易退出】===
        dynamic_margin = atr_margin_min + (atr_margin_max - atr_margin_min) * normalized

        # ===【趋势判断：若为上升趋势，保持最大margin，否则使用动态margin】===
        final_margin = np.where(df[hmm_col] == up_state, atr_margin_max, dynamic_margin)

        df["adjusted_margin"] = final_margin  # 保留用于分析，关键字：动态ATR系数

        # ===【exit_price计算】关键字：止损价、Kalman、动态平仓===
        df.loc[df.exit_price.isna(), "exit_price"] = (
            df.Kalman.shift(1) - df.ATR.shift(1) * df.adjusted_margin
        )

        return df

    def _calc_profit(self, base_df):
        last_valid_idx = base_df.exit_price.last_valid_index()
        start_idx = (
            0 if last_valid_idx is None else max(0, last_valid_idx - self.window + 1)
        )

        surf_df = base_df.iloc[start_idx:].copy()
        idx = surf_df.Stop_profit.isna()

        # 利用 tail(self.window) 中所有 High 价格计算均值和标准差（不局限于缺失行, 但结果一致）
        prices = surf_df.High.values.tolist()
        mean_price = statistics.mean(prices)
        stdv_price = statistics.stdev(prices)
        surfing_profit = mean_price + self.params.surfing_level * stdv_price

        surf_df.loc[surf_df.Stop_profit.isna(), "Stop_profit"] = surfing_profit
        # surf_df.loc[surf_df.exit_price.isna(), "exit_price"] = (
        #     surf_df.Kalman.shift(1) - surf_df.ATR.shift(1) * self.params.atr_loss_margin
        # )
        surf_df = self.adjust_exit_price_by_slope(surf_df)
        base_df.update(surf_df)

        # 第二部分：重算买卖信号与利润指标
        # 定位重新计算的起始点, 利用 sell 列缺失的第一个索引
        resume_idx = base_df.sell.isna().idxmax()
        calc_df = base_df.loc[resume_idx:].copy()

        # 买入信号：对 buy 缺失行, 赋值 Close 并调用自定义买入信号函数
        s_buy = calc_df.buy.isna()
        calc_df.loc[s_buy, "buy"] = calc_df.Close
        calc_df.loc[s_buy, "BuySignal"] = self.buy_signal_func(calc_df, self.params)

        # 卖出信号：满足已有买入信号且当日 Low 小于 exit_price 的情况
        s_sell = calc_df.buy.notna() & (calc_df.Kalman < calc_df.exit_price)
        # s_sell = calc_df.buy.notna() & (calc_df.HMM_State == 0)  # FIXME
        calc_df.loc[s_sell, "sell"] = calc_df.exit_price
        calc_df.loc[s_sell, "Matured"] = pd.to_datetime(calc_df.Date)

        # 向后填充 sell 与 Matured, 确保空缺部分得到延伸
        calc_df.sell.bfill(inplace=True)
        calc_df.Matured.bfill(inplace=True)

        # 利润计算：仅对买入与卖出均存在且 profit 缺失的行计算
        s_profit = calc_df.buy.notna() & calc_df.sell.notna() & calc_df.profit.isna()
        calc_df.loc[s_profit, "profit"] = (calc_df.sell / calc_df.buy) - 1
        calc_df.loc[s_profit, "P/L"] = (calc_df.sell - calc_df.buy) / (
            calc_df.ATR * self.params.atr_loss_margin
        )
        calc_df.loc[s_profit, "time_cost"] = [
            int(
                delta.seconds
                / 60
                / pandas_util.INTERVAL_TO_MIN.get(self.params.interval)
            )
            for delta in (
                pd.to_datetime(calc_df.loc[s_profit, "Matured"])
                - pd.to_datetime(calc_df.loc[s_profit, "Date"])
            )
        ]

        # 若 buy 为缺失, 则清空对应的 sell 和 Matured
        calc_df.loc[calc_df.buy.isna(), ["sell", "Matured"]] = [np.nan, pd.NaT]

        base_df.update(calc_df)
        return base_df

    def _calc_ATR(self, base_df):
        ATR_sample = self.params.ATR_sample
        upper_sample = self.params.upper_sample
        lower_sample = self.params.lower_sample

        last_valid_idx = base_df.ATR.last_valid_index()
        start_idx = (
            0 if last_valid_idx is None else max(0, last_valid_idx - self.window + 1)
        )

        df = base_df.iloc[start_idx:].copy()
        idx = df.ATR.isna()
        df.loc[idx, "Count_Hz"] = pandas_util.to_hz(self.params.interval, df.Count)
        df.loc[idx, "Amount_Hz"] = pandas_util.to_hz(self.params.interval, df.Amount)
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
        base_df["OBV_MA"] = df["OBV_MA"]
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

    def _calc_HMM_State(self, base_df):
        windows = self.params.bayes_windows
        mask = base_df["HMM_State"].isna()
        X = self._calc_HMM_input(base_df, windows, self.scaler)
        hidden_states = self.hmm_model.predict(X)
        base_df.loc[mask, "HMM_State"] = hidden_states[mask]
        base_df.UP_State.ffill(inplace=True)
        return base_df

    def market_recon(self, base_df):
        base_df = pandas_util.equip_fields(base_df, TURTLE_COLUMNS)
        base_df = self._calc_ATR(base_df)
        base_df = self._calc_OBV(base_df)
        base_df = calc_ADX(base_df, self.params, p=5)  # p=self.params.ATR_sample)
        base_df = self._calc_HMM_State(base_df)
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
    return (df.iloc[-1].ema_short > df.iloc[-1].ema_long) & (
        df.iloc[-1].Kalman > df.iloc[-1].ema_short
    )


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


def calc_Kalman_Price(df, windows=60, Q=1e-3, R=1e-3, alpha=0.6, gamma=0.2, beta=0.2):
    mask_lr = df["log_returns"].isna()
    df.loc[mask_lr, "log_returns"] = np.log(df["Close"] / df["Close"].shift(1))[mask_lr]

    # 计算 rolling window 的 drift（均值）和 volatility（标准差）
    drift = df["log_returns"].rolling(windows, min_periods=1).mean()
    volatility = df["log_returns"].rolling(windows, min_periods=1).std()

    mask_new = df["pred_price"].isna()
    df.loc[mask_new, "drift"] = drift[mask_new]
    df.loc[mask_new, "volatility"] = volatility[mask_new]
    df.loc[mask_new, "pred_price"] = df.loc[mask_new, "Close"] * np.exp(drift[mask_new])

    # 使用已有数据计算全局对数指标（旧数据保持不变, 新数据填充）
    df.loc[mask_new, "global_log_vol"] = (
        np.log(df["Vol"].dropna()).rolling(windows, min_periods=1).mean()
    )
    df.loc[mask_new, "global_log_volatility"] = (
        np.log(df["volatility"].dropna()).rolling(windows, min_periods=1).mean()
    )

    # 针对新数据进行 Kalman 更新（逐行计算）
    df.loc[mask_new, "Kalman"] = df.loc[mask_new].apply(
        lambda r: rolling_kalman_update(
            df,
            r.name,
            windows,
            Q,
            R,
            alpha,
            gamma,
            beta,
        ),
        axis=1,
    )
    df.loc[mask_new, "local_low"] = (
        df.Kalman.rolling(windows * 2, min_periods=1, closed="left").min().shift(1)
    )
    slope = (df.Kalman - df.local_low + epsilon) / (windows * 2)
    df.loc[mask_new, "Slope"] = slope[mask_new]
    return df


def rolling_kalman_update(df, idx, window, Q, R, alpha, gamma, beta):
    """
    从 DataFrame 中取出从 idx-window+1 到 idx 的数据构造特征矩阵,
    并调用 kalman_update 进行 Kalman 更新。
    """
    start = max(0, idx - window + 1)
    sub_df = df.loc[start:idx, ["pred_price", "Close", "volatility", "Vol"]].dropna()

    # 如果窗口内数据为空, 则返回当前行的 Close 值
    if sub_df.empty:
        return df.loc[idx, "Close"]

    # 提取窗口内的数据, 要求列顺序为：[pred_price, Close, volatility, Vol]
    features = sub_df.values
    global_log_vol = df.loc[idx, "global_log_vol"]
    global_log_volatility = df.loc[idx, "global_log_volatility"]
    return kalman_update(
        features, Q, R, alpha, gamma, beta, global_log_vol, global_log_volatility
    )


# def kalman_update(
#     features,
#     Q,
#     R_base,
#     alpha,
#     gamma,
#     global_log_vol,
#     global_log_volatility,
# ):
#     # features 的列顺序：[pred_price, Close, volatility, Vol]
#     features = features.reshape(-1, 4)
#     x = features[0, 0]
#     P = 1.0
#     for i in range(1, features.shape[0]):
#         current_vol = features[i, 3]
#         current_volatility = features[i, 2]
#         current_log_vol = np.log(current_vol)
#         current_log_volatility = (
#             np.log(current_volatility) if current_volatility > 0 else 0
#         )
#         vol_factor = np.exp(-alpha * (current_log_vol - global_log_vol))
#         volat_factor = np.exp(gamma * (current_log_volatility - global_log_volatility))
#         R_t = R_base * vol_factor * volat_factor

#         P_pred = P + Q
#         K = P_pred / (P_pred + R_t)
#         z = features[i, 1]
#         x = x + K * (z - x)
#         P = (1 - K) * P_pred
#     return x


def kalman_update(
    features,
    Q,
    R_base,
    alpha,
    gamma,
    beta,
    global_log_vol,
    global_log_volatility,
    min_vol=1e-6,
    vol_clip_min=0.1,
    vol_clip_max=2.0,
):
    """
    緊湊版卡爾曼濾波更新函數：
    結合成交量、波動率與價格變化，動態調整觀測噪聲 R_t。

    參數：
    - features: np.ndarray, 形狀 (N, 4), 欄位：[pred_price, Close, volatility, Vol]
    - Q: float, 狀態噪聲協方差
    - R_base: float, 基礎觀測噪聲協方差
    - alpha: float, 成交量調整係數
    - gamma: float, 波動率調整係數
    - beta: float, 價格變動調整係數
    - global_log_vol: float, 全球平均成交量的對數
    - global_log_volatility: float, 全球平均波動率的對數
    - min_vol: float, 成交量下限（預設 1e-6)
    - vol_clip_min: float, 成交量調整因子最小值（預設 0.1)
    - vol_clip_max: float, 成交量調整因子最大值（預設 2.0)

    回傳：
    - x: float, 濾波後的價格估計值
    """
    features = features.reshape(-1, 4)
    x = features[0, 0]  # 初始預測價格
    P = 1.0  # 初始誤差協方差
    previous_close = features[0, 1]

    for i in range(1, features.shape[0]):
        # 同時計算成交量下限與其對數
        current_log_vol = np.log(max(features[i, 3], min_vol))
        current_volatility = features[i, 2]
        current_log_volatility = (
            np.log(current_volatility) if current_volatility > 0 else 0
        )

        # 成交量調整因子（限制範圍），波動率調整因子
        vol_factor = np.clip(
            np.exp(-alpha * (current_log_vol - global_log_vol)),
            vol_clip_min,
            vol_clip_max,
        )
        volat_factor = np.exp(gamma * (current_log_volatility - global_log_volatility))

        # 價格變動因子：價格劇烈變動時，信任度降低
        current_close = features[i, 1]
        price_factor = np.exp(-beta * abs(current_close - previous_close))

        # 綜合調整後的觀測噪聲
        R_t = R_base * vol_factor * volat_factor * price_factor

        # 卡爾曼濾波標準更新步驟
        P_pred = P + Q
        K = P_pred / (P_pred + R_t)
        x = x + K * (current_close - x)
        P = (1 - K) * P_pred

        previous_close = current_close

    return x


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM


def debug_hmm(hmm_model, hidden_states, df, X):
    """
    視覺化 Hidden Markov Model (HMM) 的學習結果, 包含：
    1. 隱藏狀態時序圖
    2. 隱藏狀態高斯分佈
    3. 狀態轉移矩陣
    4. 模型對數似然值
    5. 未來狀態模擬

    參數：
    - hmm_model: 訓練好的 GaussianHMM 模型
    - df: 包含觀測數據的 DataFrame
    - feature_col: 觀測數據的欄位名稱 (預設為 "returns")
    - future_days: 模擬未來狀態的天數 (預設為 10)
    """

    ## 1️⃣ 隱藏狀態時序圖
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(df.index, df.Close, label="BTC Price", color="black", linewidth=1.5)
    ax2.plot(df.index, df.Kalman, label="Kalman Filter", color="blue", linewidth=1.5)
    ax2.legend(loc="upper right")
    ax1.set_ylabel("Features")
    ax2.set_ylabel("BTC Price")
    for i in range(hmm_model.n_components):
        ax1.fill_between(
            df.index,
            X[:, 0],
            where=(hidden_states == i),
            alpha=0.3,
            label=f"State {i} - KReturnVol",
        )
    for i in range(hmm_model.n_components):
        ax1.fill_between(
            df.index,
            X[:, 1],
            where=(hidden_states == i),
            alpha=0.3,
            label=f"State {i} - RVolume",
        )
    for i in range(hmm_model.n_components):
        ax1.fill_between(
            df.index,
            X[:, 2],
            where=(hidden_states == i),
            alpha=0.3,
            label=f"State {i} - Slope",
        )
    ax1.legend()
    plt.title("Hidden States Over Time")
    plt.xlabel("Date")
    plt.ylabel("feature_col")
    plt.show()
    ## 2️⃣ 隱藏狀態的高斯分佈
    means = hmm_model.means_.flatten()
    covars = np.sqrt(hmm_model.covars_.flatten())

    plt.figure(figsize=(8, 5))
    for i in range(hmm_model.n_components):
        sns.kdeplot(X[hidden_states == i].ravel(), label=f"State {i}", shade=True)
    plt.axvline(
        means[0], color="blue", linestyle="--", label=f"Mean State 0: {means[0]:.2f}"
    )
    plt.axvline(
        means[1], color="red", linestyle="--", label=f"Mean State 1: {means[1]:.2f}"
    )
    plt.legend()
    plt.title("Gaussian Distributions of Hidden States")
    plt.xlabel("feature_col")
    plt.show()

    ## 3️⃣ 狀態轉移矩陣（熱力圖）
    trans_mat = hmm_model.transmat_

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        trans_mat,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        xticklabels=[f"State {i}" for i in range(hmm_model.n_components)],
        yticklabels=[f"State {i}" for i in range(hmm_model.n_components)],
    )
    plt.title("HMM State Transition Matrix")
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.show()

    ## 4️⃣ 打印模型對數似然值
    log_likelihood = hmm_model.score(X)
    print(f"\n🔍 Log-Likelihood of the trained HMM: {log_likelihood:.2f}")

    print("\n✅ HMM Debug 完成！請檢查上面的圖表來分析 HMM 是否合理地劃分了市場狀態。")


from quanthunt.utils import pandas_util
from quanthunt.hunterverse.interface import IStrategyScout
from quanthunt.hunterverse.interface import IMarketSensor
from quanthunt.hunterverse.interface import IEngine
from quanthunt.hunterverse.interface import IHunter
from quanthunt.hunterverse.interface import Symbol
from quanthunt.hunterverse.interface import StrategyParam
from quanthunt.hunterverse.interface import INTERVAL_TO_MIN
from quanthunt.hunterverse.interface import xBuyOrder, xSellOrder
from quanthunt.hunterverse.interface import DEBUG_COL, DUMP_COL
from quanthunt.sensor.market_sensor import HuobiMarketSensor
from quanthunt.config.core_config import config

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir

if __name__ == "__main__":
    params = {
        # Buy
        "ATR_sample": 60,
        "bayes_windows": 10,
        "lower_sample": 60,
        "upper_sample": 60,
        # Sell
        "hard_cutoff": 0.9,
        "profit_loss_ratio": 3,
        "atr_loss_margin": 1.5,
        "surfing_level": 5,
        # Period
        "interval": "1day",
        "funds": 100,
        "stake_cap": 100,
        "symbol": None,
        "backtest": True,
        "debug_mode": [
            # "statement",
            # "statement_to_csv",
            # "mission_review",
            "final_statement_to_csv",
        ],
    }

    import click
    from typing import List

    @click.command()
    @click.option("--symbol", default="moveusdt", help="Trading symbol (e.g. trxusdt)")
    @click.option("--interval", default="1min", help="Trading interval")
    @click.option("--funds", default=50.2, type=float, help="Available funds")
    @click.option("--cap", default=10.1, type=float, help="Stake cap")
    @click.option(
        "--deals",
        default="",
        help="Comma separated deal IDs",
    )
    @click.option(
        "--start_deal",
        default=0,
        type=int,
        help="start to load from deal id",
    )
    def main(
        symbol: str,
        interval: str,
        funds: float,
        cap: float,
        deals: str,
        start_deal: int,
    ):
        """
        调试用主函数，用于测试策略参数和运行回测
        """
        deal_ids = (
            [int(x.strip()) for x in deals.split(",") if x.strip()] if deals else []
        )

        params.update(
            {
                "funds": funds,
                "stake_cap": cap,
                "symbol": Symbol(symbol),
                "interval": interval,
                "backtest": False,
                "debug_mode": [
                    "statement",
                    "statement_to_csv",
                    "mission_review",
                    "final_statement_to_csv",
                ],
                "load_deals": deal_ids,
                "start_deal": start_deal,
                "api_key": "fefd13a1-bg2hyw2dfg-440b3c64-576f2",
                "secret_key": "1a437824-042aa429-0beff3ba-03e26",
            }
        )
        sp = StrategyParam(**params)

        # 调试用代码块
        base_df = None
        load_df = pd.DataFrame()
        sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)

        # 移除 load_df 中已存在的日期部分
        update_df = sensor.scan(2000 if not sp.backtest else 100)
        if not load_df.empty:
            update_df = update_df[~update_df["Date"].isin(load_df["Date"])]

        # 合并 load_df 和 update_df，成为 base_df
        base_df = pd.concat([load_df, update_df], ignore_index=True)

        scout = TurtleScout(params=sp, buy_signal_func=emv_cross_strategy)
        import copy

        bsp = copy.deepcopy(sp)
        bsp.funds = 1000000

        base_df = sensor.scan(2000 if not sp.backtest else 100)
        base_df = scout.train(base_df)
        base_df = scout.market_recon(base_df)

        report_cols = DUMP_COL

        if "final_statement_to_csv" in sp.debug_mode:
            base_df[report_cols].to_csv(f"{REPORTS_DIR}/{sp}.csv", index=False)
            print(f"created: {REPORTS_DIR}/{sp}.csv")
        # visualize_backtest(base_df)
        return base_df[report_cols]

    main()
