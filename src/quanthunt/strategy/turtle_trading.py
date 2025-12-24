import logging
import os

import numpy as np
import pandas as pd
import statistics
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from quanthunt.strategy.algo_util.hmm_selector import HMMTrendSelector
from quanthunt.strategy.algo_util.bocpd import apply_bocpd_to_df
from quanthunt.strategy.algo_util.performance import (
    compare_signal_filters,
    analyze_hmm_states,
)
from quanthunt.strategy.algo_util.kalman import (
    init_mosaic_state,
    mosaic_step,
    build_kalman_params,
    prepare_mosaic_input,
    MosaicForceAdapter,
    MosaicPriceAdapter,
    CycleStateAdapter,
)
from quanthunt.strategy.algo_util.bocpdz import (
    BOCPDGaussianG0,
    BOCPDStudentTP1,
    DualBOCPD,
    PhaseFSMConfig,
    BOCPDPhaseFSM,
    DualBOCPDWrapper,
)
from quanthunt.hunterverse.interface import IStrategyScout, ZERO
from quanthunt.utils import pandas_util

logger = logging.getLogger(__name__)

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
    "HMM_Signal",
    "adjusted_margin",
    "Count_Hz",
    "close_sortino",
    "vol_sortino",
    "trend",
    "cv",
    "cv_sum",
    "RCS",
    "trend_norm",
    "TR_norm",
    "bias_off_norm",
    # MOSAIC model
    "force_proxy",
    "m_pc",
    "m_pt_speed",
    "m_pt_accel",
    "m_force",
    "m_force_trend",
    "m_force_bias",
    "m_z_price",
    "m_z_force",
    "m_z_mix",
    "m_regime_noise_level",
    "c_center",
    "c_z_center",
    # BOCPD G0+P1
    "bocpd_phase",
    "bocpd_cp_prob",
    "bocpd_runlen_mean",
    "bocpd_runlen_mode",
    "bocpd_risk",
    "bocpd_tail",
    "bocpd_shock",
]


class TurtleScout(IStrategyScout):
    def __init__(self, params, buy_signal_func=None):
        self.params = params
        ATR_sample = self.params.ATR_sample
        upper_sample = self.params.upper_sample
        lower_sample = self.params.lower_sample
        self.window = max(ATR_sample, upper_sample, lower_sample) * 2
        # Set the buy_signal_func, default to the simple_turtle_strategy if none is provided
        self.buy_signal_func = buy_signal_func or self._simple_turtle_strategy

        self.trend_scaler = StandardScaler()
        self.trend_hmm_model = GaussianHMM(
            n_components=self.params.hmm_split,
            covariance_type="full",
            n_iter=4000,
            random_state=42,
        )

        self.cycle_scaler = StandardScaler()
        self.cycle_hmm_model = GaussianHMM(
            n_components=self.params.hmm_split,
            covariance_type="full",
            n_iter=4000,
            random_state=42,
        )

        self.bocpd_wrapper = None
        self.kalman_state = None
        self.kalman_Cov = None
        self.m_params = None
        self.m_regime_params = None

        self.m_force_model = None
        self.m_price_model = None
        self.m_cycle_model = None

    def train(self, df):
        df = pandas_util.equip_fields(df, TURTLE_COLUMNS)
        buy_signal_from_mosaic_strategy(df, self.params)
        df = self._calc_ATR(df)
        df = self._calc_OBV(df)
        df = self.calc_kalman(df)
        df = self.train_hmm(df)
        df = self._calc_profit(df)
        return df

    def calc_kalman(self, df):
        df = prepare_mosaic_input(df)

        # Initiallize Mosaic/Cycle Kalman filter
        start_idx = df.index[df["m_pc"].isna()][0]
        if not (self.kalman_state and self.kalman_Cov):
            self.kalman_state, self.kalman_Cov = init_mosaic_state(
                init_pc=df.loc[start_idx, "Close"]
            )
            self.m_params, self.m_regime_params = build_kalman_params()
            self.m_force_model = MosaicForceAdapter(self.m_params["force"])
            self.m_price_model = MosaicPriceAdapter(self.m_params["price"])
            self.m_cycle_model = CycleStateAdapter(self.m_params["cycle"])

        for i in range(start_idx, len(df)):
            if not pd.isna(df.loc[i, "m_pc"]):
                continue

            obs_close = float(df.loc[i, "Close"])
            obs_force = float(df.loc[i, "force_proxy"])

            self.kalman_state, self.kalman_Cov, diag = mosaic_step(
                state=self.kalman_state,
                cov=self.kalman_Cov,
                obs_close=obs_close,
                obs_force=obs_force,
                force_model=self.m_force_model,
                price_model=self.m_price_model,
                cycle_model=self.m_cycle_model,
                regime_params=self.m_regime_params,
            )

            # === write back ===
            df.loc[i, "Kalman"] = np.exp(self.kalman_state["pc"])
            df.loc[i, "m_pc"] = np.exp(self.kalman_state["pc"])
            df.loc[i, "m_pt_speed"] = self.kalman_state["pt_speed"]
            df.loc[i, "m_pt_accel"] = self.kalman_state["pt_accel"]

            df.loc[i, "m_force"] = self.kalman_state["force_imbalance"]
            df.loc[i, "m_force_trend"] = self.kalman_state["force_imbalance_trend"]
            df.loc[i, "m_force_bias"] = self.kalman_state["force_proxy_bias"]

            df.loc[i, "m_z_price"] = diag["z_price"]
            df.loc[i, "m_z_force"] = diag["z_force"]
            df.loc[i, "m_z_mix"] = diag["z_mix"]
            df.loc[i, "m_regime_noise_level"] = self.kalman_state["regime_noise_level"]

            df.loc[i, "c_center"] = self.kalman_state["cycle_center"]
            df.loc[i, "c_z_center"] = diag["z_cycle"]

        return df

    def define_market_states(self, df: pd.DataFrame) -> pd.DataFrame:
        up_state = hmm_performance(df)

        s = df["UP_State"].isna()
        df.loc[s, "UP_State"] = up_state

        s = df["HMM_Signal"].isna()
        df.loc[s, "HMM_Signal"] = (
            df.loc[s, "HMM_State"] == df.loc[s, "UP_State"]
        ).astype(int)

        return df

    def train_hmm(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_trend_hmm_features(df)
        self.trend_scaler.fit(X)
        X = self.trend_scaler.transform(X)
        self.trend_hmm_model.fit(X)
        trend_hidden_states = self.trend_hmm_model.predict(X)

        Y = self._prepare_cycle_hmm_features(df)
        self.cycle_scaler.fit(Y)
        Y = self.cycle_scaler.transform(Y)
        self.cycle_hmm_model.fit(Y)
        cycle_hidden_states = self.cycle_hmm_model.predict(Y)

        # mask = df["HMM_State"].isna()
        # df.loc[mask, "HMM_State"] = trend_hidden_states[mask]

        mask = df["HMM_State"].isna()
        df.loc[mask, "HMM_State"] = cycle_hidden_states[mask]
        # df = self.define_market_states(df)
        df = self.predict_market_states(df)
        return df

    def _prepare_cycle_hmm_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        CycleHMM feature set
        - 專注於偏離、回歸動力、結構穩定度
        """

        # --- 偏離程度 ---
        df["gap"] = df["Close"] - df["c_center"]
        df["gap_norm"] = df["gap"] / df["c_z_center"].abs()

        # --- 偏離動力學 ---
        df["gap_speed"] = df["m_pc"]
        df["gap_accel"] = df["m_pt_accel"]

        # --- 結構 / regime ---
        df["z_center"] = df["c_z_center"]
        df["regime_noise"] = df["m_regime_noise_level"]
        df["z_mix"] = df["m_z_mix"]

        features = [
            "gap_norm",
            "gap_speed",
            "gap_accel",
            "z_center",
            "regime_noise",
            "z_mix",
        ]

        df[features] = df[features].fillna(method="bfill").fillna(method="ffill")
        return df[features].values

    def _prepare_trend_hmm_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        HMM 特徵工程（V4 行為 + 結構 + 趨勢方向版）

        結構類：
            - trend_norm      : 趨勢強弱（Kalman - EMA_long）/ ATR
            - TR_norm         : 波動強弱（ATR / ATR均值）的 log 比
            - slope_norm      : Kalman 斜率 / ATR（真正的趨勢方向）

        行為類：
            - RCS             : Resonance Consensus Strength（共識強度）
            - dd_proxy        : 價格自身推得的局部回撤比例
        """

        ATR_sample = self.params.ATR_sample
        ZERO = 1e-9

        # ========== 行為因子（RCS） ==========
        _momentum_factor = (df.High - df.Close) / (df.TR + ZERO)
        _res_strength = df.TR * df.Vol * _momentum_factor

        _consensus_strength = (df.Kalman / df.Close).clip(0.1, 10) * df.Vol
        _consensus_norm = (
            _consensus_strength / _consensus_strength.rolling(window=ATR_sample).mean()
        )

        df["RCS"] = _res_strength * _consensus_norm

        # ========== 結構因子 ==========
        # 趨勢強弱（Kalman - EMA_long）
        df["trend"] = df.Kalman - df["ema_long"]
        df["trend_norm"] = df.trend / (df.ATR + ZERO)

        # 波動強弱（ATR 相對於自身歷史均值）
        df["TR_norm"] = np.log(
            df.ATR + ZERO / (df.ATR.rolling(window=ATR_sample).mean() + ZERO)
        )

        # **新增：趨勢方向（Kalman 斜率）**
        df["slope_norm"] = df.m_pt_speed.diff() / (df.ATR + ZERO)

        # ========== 新增：趨勢持續性 / 偏態 / 回撤 Proxy ==========
        # 3) 局部 Drawdown proxy
        roll_max = df["Close"].cummax()
        df["dd_proxy"] = (roll_max - df["Close"]) / (roll_max + ZERO)

        # --- Final features ---
        features = [
            "trend_norm",
            "slope_norm",
            "RCS",
            "TR_norm",
            "dd_proxy",
        ]

        df[features] = df[features].fillna(method="bfill").fillna(method="ffill")
        return df[features].values

    def predict_market_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根據 HMMTrendSelector 選出「最值得做多的 HMM_State」
        並填入：
            - UP_State  : 全局推薦的上升狀態（單一 state id）
            - HMM_Signal: 當前 row 的 HMM_State == UP_State 時，給 1，否則 0
        """

        # 確保 profit 欄位存在（這裡假設你回測已經算好）
        if "profit" not in df.columns:
            raise ValueError(
                "define_market_states 需要 df 內含 'profit' 欄位，請先回填策略盈虧。"
            )

        # 1) 用 Trend μ 挑出最強的單一 state
        selector = HMMTrendSelector(
            df,
            state_col="HMM_State",
            profit_col="profit",
            min_samples=500,  # 可改成 self.params.min_hmm_samples 之類
        )
        best_states = selector.best_states(top_n=1)
        best_combo = selector.best_combos(top_n=1)
        combo_states = set(best_combo.iloc[0]["combo"])
        print(selector.best_combos(top_n=1))
        if best_states.empty or best_states["trend_mu"].iloc[0] <= 0:
            # 找不到有正期望的 state，就不要亂打 HMM_Signal
            return df

        up_state = int(best_states["state"].iloc[0])

        # 2) 只填尚未設定的 UP_State
        s = df["UP_State"].isna()
        df.loc[s, "UP_State"] = up_state

        # 3) HMM_Signal：當前 HMM_State == UP_State 時標記 1
        s = df["HMM_Signal"].isna()
        # Single HMM State
        # df.loc[s, "HMM_Signal"] = (
        #     df.loc[s, "HMM_State"] == df.loc[s, "UP_State"]
        # ).astype(int)
        # Combo HMM State
        df.loc[s, "HMM_Signal"] = df.loc[s, "HMM_State"].isin(combo_states).astype(int)
        return df

    def predict_hmm(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare_trend_hmm_features(df)
        X = self.trend_scaler.transform(X)
        trend_hidden_states = self.trend_hmm_model.predict(X)

        Y = self._prepare_cycle_hmm_features(df)
        Y = self.cycle_scaler.transform(Y)
        cycle_hidden_states = self.cycle_hmm_model.predict(Y)

        # mask = df["HMM_State"].isna()
        # df.loc[mask, "HMM_State"] = trend_hidden_states[mask]

        mask = df["HMM_State"].isna()
        df.loc[mask, "HMM_State"] = cycle_hidden_states[mask]

        # df = self.define_market_states(df)
        df = self.predict_market_states(df)
        return df

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
        hmm_signal = df.iloc[-1].HMM_Signal  # 关键字：HMM、上升趋势状态
        slope_col = "m_pt_speed"
        atr_margin_max = self.params.atr_loss_margin
        atr_margin_min = getattr(self.params, "atr_margin_min", 0.1)

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
        final_margin = np.where(hmm_signal == 1, atr_margin_max, dynamic_margin)

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
        df.loc[idx, "turtle_h"] = df.High.shift(1).rolling(upper_sample).max()
        df.loc[idx, "turtle_l"] = df.Low.shift(1).rolling(lower_sample).min()
        df.loc[idx, "h_l"] = df.High - df.Low
        df.loc[idx, "c_h"] = (df.Close.shift(1) - df.High).abs()
        df.loc[idx, "c_l"] = (df.Close.shift(1) - df.Low).abs()
        df.loc[idx, "TR"] = df[["h_l", "c_h", "c_l"]].max(axis=1)
        df.loc[idx, "ATR"] = df["TR"].rolling(ATR_sample).mean()
        df.loc[idx, "ATR_STDV"] = df["TR"].rolling(ATR_sample).std()

        df.loc[idx, "cv"] = np.log(df.TR * df.Vol)
        df.loc[idx, "cv_sum"] = df.cv.rolling(window=ATR_sample).sum()

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

    def calc_bocpd(self, df):
        if not self.bocpd_wrapper:
            # --- init once ---
            g0 = BOCPDGaussianG0(hazard=0.01, r_max=300)
            p1 = BOCPDStudentTP1(hazard=0.01, r_max=300)

            dual = DualBOCPD(g0, p1)
            cfg = PhaseFSMConfig()
            phase_fsm = BOCPDPhaseFSM(cfg)
            phase_fsm.warmup_ticks = 400

            self.bocpd_wrapper = DualBOCPDWrapper(
                dual_bocpd=dual,
                phase_fsm=phase_fsm,
                x_col="m_z_mix",
            )
            self.bocpd_wrapper.run_online(df)
        else:
            self.bocpd_wrapper.update_df_row(df, df.index[-1])
        return df

    def market_recon(self, base_df):
        base_df = pandas_util.equip_fields(base_df, TURTLE_COLUMNS)
        buy_signal_from_mosaic_strategy(base_df, self.params)
        base_df = self._calc_ATR(base_df)
        base_df = self._calc_OBV(base_df)
        base_df = calc_ADX(base_df, self.params, p=5)  # p=self.params.ATR_sample)
        base_df = self.calc_kalman(base_df)
        base_df = self.calc_bocpd(base_df)
        base_df = self.predict_hmm(base_df)
        base_df = self._calc_profit(base_df)
        # HACK remove me
        print(compare_signal_filters(base_df))
        print(analyze_hmm_states(base_df))
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
    return (df.Kalman > df.ema_short) & (df.ema_short > df.ema_long)
    # return (df.iloc[-1].ema_short > df.iloc[-1].ema_long) & (
    #     df.iloc[-1].Kalman > df.iloc[-1].ema_short
    # )


def buy_signal_from_mosaic_strategy(df, params, short_windows=5, long_windws=60):
    """
    BuySignal = 結構位置 + 力學轉折（threshold 隨 regime 調整）
    假設 HMM 已在外層 gate 過
    """

    # === EMA（保留，雖然目前沒直接用在 score）===
    idx = df.ema_short.isna()
    df.loc[idx, "ema_short"] = df.Close.ewm(span=short_windows, adjust=False).mean()
    df.loc[idx, "ema_long"] = df.Close.ewm(span=long_windws, adjust=False).mean()

    # === 結構條件 ===
    cond_structure = df.c_z_center < -1.0

    # === 力學條件 ===
    cond_force_pos = df.m_force_trend > 0
    cond_force_stable = df.m_force_trend.rolling(20).quantile(0.3) > 0

    # === 世界狀態（regime noise）===
    regime_noise = df.m_regime_noise_level
    regime_mean = regime_noise.rolling(20).mean()

    # === regime 分段（低 / 中 / 高噪音）===
    low_noise = regime_noise < regime_mean * 0.9
    mid_noise = (regime_noise >= regime_mean * 0.9) & (
        regime_noise <= regime_mean * 1.1
    )
    high_noise = regime_noise > regime_mean * 1.1

    # === Buy score（你原本的設計，保留）===
    buy_score = 1.25 * cond_force_pos + 1.25 * cond_force_stable
    # buy_score = 1.0 * cond_structure + 0.5 * cond_force_pos + 0.5 * cond_force_stable

    # === Regime-adaptive threshold ===
    threshold = (
        1.0 * low_noise  # 世界乾淨：一個強理由就可以
        + 1.5 * mid_noise  # 世界普通：至少 1.5 分
        + 2.0 * high_noise  # 世界混亂：幾乎要全對
    )

    return buy_score >= threshold


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


def hmm_standardize(series: pd.Series, window: int = 20) -> pd.DataFrame:
    _log = np.log(series + ZERO)
    _ma = series.rolling(window).mean()
    _relative_strength = series / _ma
    _zscore = (series - _ma) / series.rolling(window).std()
    # _sortino = sortino_ratio(series)
    return _log, _ma, _relative_strength, _zscore


def sortino_ratio(series, rf=0.0):
    downside = series[series < rf]
    downside_std = downside.std(ddof=0)
    if np.isnan(downside_std):
        downside_std = ZERO
    return (series.mean() - rf) / downside_std


def hmm_performance(df: pd.DataFrame, min_count: int = 300) -> int:
    """
    原理（直覺版）：
    1) HMM 每個 state 代表一種「市場狀態」；我們不想被樣本太少的 state 騙。
    2) 先用 state 的出現次數過濾掉「冷門/不可靠」狀態（count >= min_count）。
    3) 對每個可靠 state 取特徵均值，跟「理想的核心趨勢狀態」做距離比較：
       - consensus_norm 越高越好（+1）
       - trend_norm 越高越好（+1）
       - TR_norm 越接近 0 越好（0，代表不過熱不過冷）
       - bias_off_norm 越低越好（-1）
    4) 距離越小越像核心趨勢狀態；回傳該 state id。
    """
    df = df.copy()

    counts = df["HMM_State"].value_counts()
    valid_states = counts[counts >= min_count].index

    prof = (
        df.groupby("HMM_State")[["RCS", "trend_norm", "TR_norm", "bias_off_norm"]]
        .mean()
        .loc[valid_states]
    )

    ideal = pd.Series(
        {
            "RCS": +1.0,
            "trend_norm": +1.0,
            "TR_norm": 0.0,
            "bias_off_norm": -1.0,
        }
    )

    prof["score"] = (prof.sub(ideal) ** 2).sum(axis=1)
    print(prof)
    return int(prof["score"].idxmin())


# def hmm_performance(df: pd.DataFrame):
#     df = df.copy()
#     df["log_return"] = np.log(df.Close / df.Close.shift(1))
#     df.dropna(subset=["log_return", "HMM_State"], inplace=True)
#     result = (
#         df.groupby("HMM_State")
#         .agg(
#             avg_return=("log_return", "mean"),
#             std_return=("log_return", "std"),
#             count=("log_return", "count"),
#             sortino_ratio=("log_return", sortino_ratio),
#         )
#         .reset_index()
#     )

#     return result.sortino_ratio.idxmax()


# def hmm_performance(df: pd.DataFrame):
#     df = df.copy()

#     # 計算 log return
#     df["log_return"] = np.log(df.Close / df.Close.shift(1))

#     # 移除 NaN
#     df.dropna(subset=["log_return", "HMM_State"], inplace=True)

#     # 建立輸出表格
#     result = (
#         df.groupby("HMM_State")
#         .agg(
#             avg_return=("log_return", "mean"),
#             std_return=("log_return", "std"),
#             count=("log_return", "count"),
#             sortino_ratio=("log_return", sortino_ratio),
#         )
#         .reset_index()
#     )

#     # 計算平均持續時間（每段連續出現狀態的長度）
#     durations = []
#     prev_state = None
#     count_duration = 0
#     for state in df["HMM_State"]:
#         if state == prev_state:
#             count_duration += 1
#         else:
#             if prev_state is not None:
#                 durations.append((prev_state, count_duration))
#             prev_state = state
#             count_duration = 1
#     if prev_state is not None:
#         durations.append((prev_state, count_duration))

#     duration_df = pd.DataFrame(durations, columns=["state", "duration"])
#     avg_durations = (
#         duration_df.groupby("state")["duration"]
#         .mean()
#         .reset_index()
#         .rename(columns={"state": "HMM_State", "duration": "avg_duration"})
#     )

#     # 合併兩張表
#     result = result.merge(avg_durations, on="HMM_State", how="left")

#     # 加入 weighted score 計算（正規化方式）
#     result["weight_count_std"] = (result["count"] / result["std_return"]).rank(pct=True)
#     result["weight_duration"] = np.sqrt(result["avg_duration"]).rank(pct=True)
#     result["weighted_score"] = result["sortino_ratio"] * (
#         0.6 * result["weight_count_std"] + 0.4 * result["weight_duration"]
#     )
#     return result


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
    import click

    @click.command()
    @click.option("--symbol", default="moveusdt", help="Trading symbol (e.g. trxusdt)")
    @click.option("--interval", default="1min", help="Trading interval")
    @click.option("--count", default=2000, help="load datas")
    @click.option("--hmm_split", default=5, type=int, help="hmm status split")
    def main(symbol: str, interval: str, count: int, hmm_split: int):
        params = {
            "ATR_sample": 60,
            "bayes_windows": 20,
            "lower_sample": 60,
            "upper_sample": 60,
            "hard_cutoff": 0.9,
            "profit_loss_ratio": 3,
            "atr_loss_margin": 1.5,
            "surfing_level": 5,
            "interval": interval,
            "funds": 50,
            "stake_cap": 10,
            "symbol": Symbol(symbol),
            "hmm_split": hmm_split,
            "backtest": True,
            "debug_mode": ["statement"],
            "api_key": os.getenv("API_KEY"),
            "secret_key": os.getenv("SECRET_KEY"),
        }
        sp = StrategyParam(**params)

        sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)
        base_df = sensor.scan(count)
        # cut = pd.Timestamp("2025-06-04 15:00:00")
        # base_df = base_df[base_df["Date"] >= cut].reset_index()

        scout = TurtleScout(params=sp, buy_signal_func=buy_signal_from_mosaic_strategy)
        scout = TurtleScout(sp)
        base_df = scout.train(base_df)
        base_df = scout.market_recon(base_df)
        report_cols = DUMP_COL

        base_df[report_cols].to_csv(f"{REPORTS_DIR}/{sp}_hmm_test.csv", index=False)
        print(f"created: {REPORTS_DIR}/{sp}_hmm_test.csv")
        hmm_table = hmm_performance(base_df)
        # print(hmm_table)
        print(
            f"BEST HMM_STATE: {hmm_table.loc[hmm_table['weighted_score'].idxmax(), 'HMM_State']}"
        )
        return base_df[report_cols]

    main()
