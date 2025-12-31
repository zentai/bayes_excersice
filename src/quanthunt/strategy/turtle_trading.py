import logging
import os

import numpy as np
import pandas as pd
import statistics
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from quanthunt.strategy.algo_util.hmm_selector import HMMTrendSelector
from quanthunt.strategy.algo_util.performance import (
    compare_performance,
    compare_signal_filters,
    analyze_hmm_states,
    evaluate_hmm_signal,
    train_test_split_by_time,
)
from quanthunt.strategy.algo_util.kalman import (
    init_kalman_state,
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
    "buy",
    "sell",
    "profit",
    "time_cost",
    "Matured",
    "BuySignal",
    "P/L",
    "ema_short",
    "ema_long",
    "Kalman",
    "HMM_State",
    "HMM_Epoch",
    "HMM_Signal",
    "adjusted_margin",
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

        self.scaler = StandardScaler()
        self.hmm_model = GaussianHMM(
            n_components=self.params.hmm_split,
            covariance_type="full",
            n_iter=4000,
            random_state=42,
            init_params="",  # ❗不要覆蓋你的手動初始化
            params="mcst",  # 訓練時仍可更新 mean/cov/start/trans
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
        df = self._calc_ATR(df)
        df = self.calc_kalman(df)
        df = self._calc_profit(df)
        df = self.train_hmm(df)  # HMM_Signal must based on Profits
        return df

    def calc_kalman(self, df):
        df = prepare_mosaic_input(df)

        # Initiallize Mosaic/Cycle Kalman filter
        start_idx = df.index[df["m_pc"].isna()][0]
        if not (self.kalman_state and self.kalman_Cov):
            self.kalman_state, self.kalman_Cov = init_kalman_state(
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

    def train_hmm(self, df: pd.DataFrame, reset=False) -> pd.DataFrame:
        """
        完整 HMM retrain（通常在 regime reset 後觸發）
        重啟「世界語意」，但不回頭覆蓋歷史 row

        設計理念：
            - reset 代表世界觀更新
            - 產生新的 HMM_Epoch
            - 但歷史的 profit / signal / state 都不覆寫（避免前視偏差）

        運作流程：
            1) 對全資料 fit scaler + fit HMM（Trend & Cycle）
            2) 只更新 HMM_State 為 NaN 的 row（之前未觀測過）
            3) Shadow DF 全量評估 → 產生新的 best combo
            4) 記錄新的 HMM_Epoch
        """

        # === 0) HMM_Epoch 遞增策略 ===
        if df["HMM_Epoch"].isna().all():
            df["HMM_Epoch"] = 0

        # === 1) Trend HMM retrain ===
        if reset:
            self.hmm_model = GaussianHMM(
                n_components=self.params.hmm_split,
                covariance_type="full",
                n_iter=4000,
                random_state=42,
                init_params="",  # ❗不要覆蓋你的手動初始化
                params="mcst",  # 訓練時仍可更新 mean/cov/start/trans
            )

            mask = df["HMM_State"].isna()  # 僅限當前要補的 row（新資料）
            df.loc[mask, "HMM_Epoch"] = df["HMM_Epoch"].max() + 1

        if self.params.hmm_model == "trend":
            X = self._prepare_trend_hmm_features(df)
        else:
            X = self._prepare_cycle_hmm_features(df)

        self.scaler.fit(X)
        X = self.scaler.transform(X)

        self.hmm_model.fit(X)
        hidden_states = self.hmm_model.predict(X)

        # === 3) 只補上「尚未被填過」的 row ===
        # 歷史語意不可回頭重寫！
        mask = df["HMM_State"].isna()
        df.loc[mask, "HMM_State"] = hidden_states[mask]

        # === 4) Shadow DF：採用最新語意世界來挑 best combo ===
        _shadow_df = df.copy()
        _shadow_df["HMM_State"] = hidden_states

        combo_states = self.select_profitable_states(_shadow_df)
        # === 5) 只更新未設定的信號 ===
        s = df["HMM_Signal"].isna()
        if combo_states:
            df.loc[s, "HMM_Signal"] = (
                df.loc[s, "HMM_State"].isin(combo_states).astype(int)
            )
        else:
            df.loc[s, "HMM_Signal"] = 0
        return df

    def _prepare_cycle_hmm_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        CycleHMM feature set
        - 專注於偏離、回歸動力、結構穩定度
        """
        print(f"Working with Cycle HMM Feature")

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
            "gap_speed",
            "gap_accel",
            "regime_noise",
            "z_mix",
            "z_center",
            "gap_norm",
        ]

        df[features] = df[features].fillna(method="bfill").fillna(method="ffill")

        # # === Debug Block ===
        # Y = df[features].values
        # bad_mask = ~np.isfinite(Y)

        # if bad_mask.any():
        #     rows, cols = np.where(bad_mask)
        #     print("⚠️ CycleHMM Feature Data Issue Detected:")
        #     for r, c in zip(rows[:10], cols[:10]):  # 限制输出10条避免爆屏
        #         print(f" - Row {r}, Feature: {features[c]}, Value: {Y[r, c]}")
        #     print(f"Total bad entries: {bad_mask.sum()}")
        #     print(f"Columns summary:")
        #     for i, f in enumerate(features):
        #         cnt = bad_mask[:, i].sum()
        #         if cnt:
        #             print(f"   {f}: {cnt} bad values")

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
        print(f"Working with Trend HMM Feature")
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

        # === EMA（保留，雖然目前沒直接用在 score）===
        idx = df.ema_short.isna()
        df.loc[idx, "ema_short"] = df.Close.ewm(span=5, adjust=False).mean()
        df.loc[idx, "ema_long"] = df.Close.ewm(span=60, adjust=False).mean()

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

    def select_profitable_states(self, df: pd.DataFrame) -> pd.DataFrame:
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

        if best_states.empty or best_states["trend_mu"].iloc[0] <= 0:
            print("# 找不到有正期望的 state，就不要亂打 HMM_Signal")
            return set()

        # print(selector.best_combos(top_n=1))
        return combo_states

    def predict_hmm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        線上預測 HMM，目前基於 cycle HMM（可自由切換 hybrid）
        本函式做兩件事：
            1) 僅在 HMM_State 為 NaN 的 row 上補上預測值（避免覆蓋歷史）
            2) 用 Shadow Copy 重新評估所有 state，挑出最佳 state combo 並設定當前信號

        設計理念：
            「交易過程中，HMM 會多次 reset，state 語意會變動」
            因此：
            - 真實 df：維持 backtest 過程的「一致性歷史語意」
            - Shadow df：允許使用全部資料重新推論當前最佳狀態
            - 只將「結果中的信號」回寫到真實 df
            - 避免因為覆寫舊 state 而產生前視偏差 (look-ahead bias)
        """

        # === 1) Trend HMM 預測 (目前不更新主 df，只保留參考 ===
        if self.params.hmm_model == "trend":
            X = self._prepare_trend_hmm_features(df)
        else:
            X = self._prepare_cycle_hmm_features(df)

        X = self.scaler.transform(X)
        hidden_states = self.hmm_model.predict(X)

        # === 2) Cycle HMM 預測：本次實際使用 ===

        # === 3) 僅在 HMM_State 為 NaN 時補寫預測結果 ===
        #    讓回測的歷史 state 不會被重算覆寫

        mask = df["HMM_State"].isna()
        df.loc[mask, "HMM_State"] = hidden_states[mask]

        # === 4) Shadow DF：完整新預測，不覆蓋主 df ===
        #    用於「當下最佳 HMM combo」選擇，不污染歷史語意
        _shadow_df = df.copy()
        _shadow_df["HMM_State"] = hidden_states

        # === 5) 用 Shadow DF 做策略最佳狀態決策 ===
        combo_states = self.select_profitable_states(_shadow_df)

        # === 6) 僅在「未決定過信號」的 row 上更新 HMM_Signal ===
        #    避免 refit 後回頭修改歷史訊號，導致回測失真
        s = df["HMM_Signal"].isna()
        if combo_states:
            df.loc[s, "HMM_Signal"] = (
                df.loc[s, "HMM_State"].isin(combo_states).astype(int)
            )
        else:
            df.loc[s, "HMM_Signal"] = 0
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
        # calc_df = base_df.loc[resume_idx:].copy()
        calc_df = base_df.copy()

        # 买入信号：对 buy 缺失行, 赋值 Close 并调用自定义买入信号函数
        s_buy = calc_df.buy.isna()
        calc_df.loc[s_buy, "buy"] = calc_df.Close
        calc_df.loc[s_buy, "BuySignal"] = self.buy_signal_func(calc_df, self.params)[
            s_buy
        ]

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
        df.loc[idx, "turtle_h"] = df.High.shift(1).rolling(upper_sample).max()
        df.loc[idx, "turtle_l"] = df.Low.shift(1).rolling(lower_sample).min()
        df.loc[idx, "h_l"] = df.High - df.Low
        df.loc[idx, "c_h"] = (df.Close.shift(1) - df.High).abs()
        df.loc[idx, "c_l"] = (df.Close.shift(1) - df.Low).abs()
        df.loc[idx, "TR"] = df[["h_l", "c_h", "c_l"]].max(axis=1)
        df.loc[idx, "ATR"] = df["TR"].rolling(ATR_sample).mean()

        base_df.update(df)
        return base_df

    def calc_bocpd(self, df):
        if not self.bocpd_wrapper:
            # --- init once ---
            g0 = BOCPDGaussianG0(hazard=0.015, r_max=300)
            p1 = BOCPDStudentTP1(hazard=0.05, r_max=300)

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
        base_df = self._calc_ATR(base_df)
        base_df = self.calc_kalman(base_df)
        base_df = self.calc_bocpd(base_df)
        if base_df.iloc[-1].bocpd_runlen_mode == 0:
            base_df = self.train_hmm(base_df, reset=True)
        else:
            base_df = self.predict_hmm(base_df)
        base_df = self._calc_profit(base_df)
        # HACK remove me
        # print(compare_signal_filters(base_df))
        # print(analyze_hmm_states(base_df))

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


def buy_signal_from_mosaic_strategy(df: pd.DataFrame, params) -> pd.DataFrame:

    mask = df["BuySignal"].isna()

    # === A) 感知市場換擋 ===
    noise_up = df["m_regime_noise_level"].diff() > 0.6
    noise_drop = df["m_regime_noise_level"] < 2.0

    # === B) 力量確認 ===
    trend_ok = df["m_force_trend"].rolling(3).mean() > 0
    bias_ok = df["m_force_bias"] > 0

    # === C) 價格確認 ===
    # 近7根新高視為突破
    price_up = df["Close"] > df["Close"].rolling(7).max().shift(1)

    # Phase Shift → Confirm → Entry
    # buy_cond = (
    #     noise_up.shift(1).fillna(False) & noise_drop & trend_ok & bias_ok & price_up
    # )
    buy_cond = noise_drop & trend_ok & bias_ok & price_up

    df.loc[mask & buy_cond, "BuySignal"] = 1
    df.loc[mask & ~buy_cond, "BuySignal"] = 0

    return df


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM


def sortino_ratio(series, rf=0.0):
    downside = series[series < rf]
    downside_std = downside.std(ddof=0)
    if np.isnan(downside_std):
        downside_std = ZERO
    return (series.mean() - rf) / downside_std


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

# if __name__ == "__main__":
#     import click

#     @click.command()
#     @click.option("--symbol", default="moveusdt", help="Trading symbol (e.g. trxusdt)")
#     @click.option("--interval", default="1min", help="Trading interval")
#     @click.option("--count", default=2000, help="load datas")
#     @click.option("--hmm_split", default=5, type=int, help="hmm status split")
#     def main(symbol: str, interval: str, count: int, hmm_split: int):
#         params = {
#             "ATR_sample": 60,
#             "bayes_windows": 20,
#             "lower_sample": 60,
#             "upper_sample": 60,
#             "hard_cutoff": 0.9,
#             "profit_loss_ratio": 3,
#             "atr_loss_margin": 1.5,
#             "surfing_level": 5,
#             "interval": interval,
#             "funds": 50,
#             "stake_cap": 10,
#             "symbol": Symbol(symbol),
#             "hmm_split": hmm_split,
#             "backtest": True,
#             "debug_mode": ["statement"],
#             "api_key": os.getenv("API_KEY"),
#             "secret_key": os.getenv("SECRET_KEY"),
#         }
#         sp = StrategyParam(**params)

#         sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)
#         base_df = sensor.scan(count)
#         # cut = pd.Timestamp("2025-06-04 15:00:00")
#         # base_df = base_df[base_df["Date"] >= cut].reset_index()

#         scout = TurtleScout(params=sp, buy_signal_func=buy_signal_from_mosaic_strategy)
#         scout = TurtleScout(sp)
#         base_df = scout.train(base_df)
#         base_df = scout.market_recon(base_df)
#         report_cols = DUMP_COL

#         base_df[report_cols].to_csv(f"{REPORTS_DIR}/{sp}_hmm_test.csv", index=False)
#         print(f"created: {REPORTS_DIR}/{sp}_hmm_test.csv")
#         return base_df[report_cols]

#     main()


# def main():
#     # ===== Futures Candidate List =====
#     # 第一批
#     # "GC=F", "CL=F", "HG=F", "ZN=F"

#     # 第二批（部分 OUT）
#     # "SI=F", "NG=F", "ZW=F", "ZS=F", "ZC=F"

#     # 第三批
#     # "ES=F", "6E=F", "6J=F"

#     symbols = [
#         "GC=F",
#         "CL=F",
#         "HG=F",
#         "ZN=F",
#         "SI=F",
#         "NG=F",
#         "ZW=F",
#         "ZS=F",
#         "ZC=F",
#         "ES=F",
#         "6E=F",
#         "6J=F",
#         "SOLX",  # 你原本有加入的案例
#     ]

#     # === Future optional tests (保留你的註解) ===
#     # FX:
#     # "NEAR-USD", "PG", "JNJ", "PLTR", "TSLA", "BAC", "VOO", "G13.SI", "TON-USD", "ADBE"

#     # Worst test cases:
#     # "FX"
#     # "ARKK"
#     # "NVDA", "GM"
#     # "BABA"
#     # "QQQ"
#     # "VOO"
#     # "MCD"
#     # "KO"

#     for symbol in symbols:
#         print(f"\n===== {symbol} =====")
#         result_df = run(symbol)
#         print(result_df)


if __file__ == "__main__":
    params = {
        "ATR_sample": 60,
        "bayes_windows": 20,
        "lower_sample": 60,
        "upper_sample": 60,
        "hard_cutoff": 0.975,
        "profit_loss_ratio": 3,
        "atr_loss_margin": 1,
        "surfing_level": 5,
        "interval": "1min",
        "funds": 50,
        "stake_cap": 10,
        "hmm_split": 3,
        "backtest": True,
        "debug_mode": ["statement"],
        "api_key": "",
        "secret_key": "",
    }

    sp = StrategyParam(**params)

    # ===== Data Loading =====
    base_df = pd.read_csv(f"/Users/zen/Documents/code/bayes/data/btcusdt_cached.csv")
    base_df["Date"] = pd.to_datetime(base_df["Date"])
    base_df = base_df.reset_index(drop=True)

    # ===== Split train & test =====
    top_10pct = int(len(base_df) * 0.55)
    train_df = base_df.iloc[:top_10pct].copy()
    test_df = base_df.iloc[top_10pct:].copy()
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    test_df["Matured"] = pd.NaT

    print(f"PROCESS table size: {len(base_df)}")

    # ===== Training Phase =====
    scout = TurtleScout(param=sp, buy_signal_func=emv_cross_strategy)
    base_df = scout.train(train_df)
    base_df = scout.market_recon(base_df)
    _train = base_df.copy()
    if not os.path.exists(REPORTS_DIR):
        os.mkdir(REPORTS_DIR)

    update_idx = 0

    # ===== Step-by-step Online Backtest =====
    for _ in range(len(test_df)):
        new_row = test_df.iloc[update_idx].copy()
        new_row["Matured"] = pd.NaT
        new_row["Date"] = pd.to_datetime(new_row["Date"])

        new_df = pd.DataFrame([new_row], columns=base_df.columns)
        base_df = pd.concat([base_df, new_df], ignore_index=True)

        # 市場重建
        base_df = scout.market_recon(base_df)

        # Output in debug
        print(base_df[DUMP_COL])

        update_idx += 1

    _test = base_df.iloc[top_10pct:].copy()
    _train_stats, _test_stats = compare_performance(_train, _test)
    print(_train_stats)
    print(_test_stats)
