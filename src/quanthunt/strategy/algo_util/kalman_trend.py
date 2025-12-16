import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from plot_performance import (
    check_sample_size,
    plot_strategy_kde_comparison,
    plot_strategy_cdf_comparison,
    plot_kl_matrix,
    plot_tail_quadrant,
    simulate_profit_by_kde,
)


def compute_atr(df, atr_window=14):
    # ------------------------------
    # 0) 計算 ATR 與 range_expansion (TF3)
    # ------------------------------
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(atr_window).mean()
    df["ATR"] = atr.bfill()
    return df


def kalman_close_3rd(
    df,
    window=200,
    qL=1e-5,
    qS=1e-4,
    qA=1e-3,
    r0=1e-2,
    a1=0.5,
    a2=0.5,
    b1=0.3,
    b2=0.5,
    b3=0.3,
):
    """
    df: 需有 ['Open','High','Low','Close','Vol']
    回傳: 新欄位 ['kc_level','kc_slope','kc_accel']
    """
    df = df.copy()
    # 1) 計算 feature
    df = compute_atr(df)  # e.g. 14 or 20
    df["ATR_mean"] = df["ATR"].rolling(window).mean()
    df["ATR_norm"] = (df["ATR"] / df["ATR_mean"]).clip(0.5, 3.0)
    df["range_exp"] = (df["High"] - df["Low"]) / (df["ATR"] + 1e-9)
    df["range_exp"] = df["range_exp"].clip(0, 5.0)
    df["log_vol"] = np.log(df["Vol"].replace(0, np.nan)).fillna(0)
    df["log_vol_mean"] = df["log_vol"].rolling(window).mean()
    df["log_vol_std"] = df["log_vol"].rolling(window).std().replace(0, 1)
    df["volume_z"] = (df["log_vol"] - df["log_vol_mean"]) / df["log_vol_std"]
    df = df.bfill()  # HACK?

    # 2) 初始化
    n = len(df)
    level = np.zeros(n)
    slope = np.zeros(n)
    accel = np.zeros(n)

    F = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    I = np.eye(3)
    Q0 = np.diag([qL, qS, qA])

    # 初始 state & covariance
    x = np.array([df["Close"].iloc[0], 0.0, 0.0])
    P = np.eye(3)

    for t in range(n):
        close_t = df["Close"].iloc[t]

        ATR_norm_t = df["ATR_norm"].iloc[t]
        range_exp_t = df["range_exp"].iloc[t]
        volume_z_t = df["volume_z"].iloc[t]

        # 2a) 動態 Q_t
        vol_factor = np.exp(a1 * (ATR_norm_t - 1.0))
        range_factor = np.exp(a2 * (range_exp_t - 1.0))
        Q_t = Q0 * vol_factor * range_factor

        # 2b) 動態 R_t
        vol_factor_R = np.exp(-b1 * volume_z_t)
        volat_factor_R = np.exp(b2 * (ATR_norm_t - 1.0))
        range_factor_R = np.exp(b3 * (range_exp_t - 1.0))
        R_t = r0 * vol_factor_R * volat_factor_R * range_factor_R

        # 3) predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q_t

        # 4) update (scalar measurement)
        y_pred = x_pred[0]
        S = P_pred[0, 0] + R_t
        K = P_pred[:, 0] / S
        innov = close_t - y_pred
        x = x_pred + K * innov
        P = (I - np.outer(K, np.array([1.0, 0.0, 0.0]))) @ P_pred

        level[t], slope[t], accel[t] = x
    df["kc_level"] = level
    df["kc_slope"] = slope
    df["kc_accel"] = accel
    print(df[["Date", "Close", "kc_level"]])

    return df


def _compute_force_proxies(df, eps=1e-9):
    """
    從 OHLCV 計算兩個力量投影：
    - HL_ret: 幾何方向 (Close-Open)/(High-Low)
    - pseudo_delta: HL_ret * Vol (力量密度)
    這裡會做 z-score 標準化，避免 pseudo_delta 尺度爆炸。
    期待 df.columns 至少有: ['Open','High','Low','Close','Vol']
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    open_ = df["Open"]
    vol = df["Vol"]

    rng = (high - low).replace(0, np.nan)

    hl_ret = (close - open_) / (rng + eps)
    pseudo_delta = hl_ret * vol

    proxies = pd.DataFrame(
        {
            "HL_ret": hl_ret,
            "pseudo_delta": pseudo_delta,
        },
        index=df.index,
    )

    # 標準化：每個 proxy 做 z-score
    for col in proxies.columns:
        m = proxies[col].mean()
        s = proxies[col].std()
        proxies[col] = (proxies[col] - m) / (s + eps)

    return proxies.fillna(0.0)


def kalman_trend_force_multi_obs(df, window=200, Q=None, R=None, init_state=None):
    """
    多觀測版 3 階 Kalman + rolling window

    觀測 y_t = [HL_ret_t, pseudo_delta_t]^T
    狀態 x_t = [Level_t, Slope_t, Accel_t]^T

    df: DataFrame，至少包含欄位:
        - 'Open', 'High', 'Low', 'Close', 'Vol'
    window: 每次估計時使用的滾動視窗長度 (bar 數)

    回傳: 新的 DataFrame，增加欄位：
        - 'Ret'           : Close-to-Close return
        - 'HL_ret'
        - 'pseudo_delta'
        - 'level'
        - 'slope'
        - 'accel'
    """
    df = df.copy()

    # 基本 return（保留給之後其他用途）
    df["Ret"] = df["Close"].pct_change().fillna(0.0)

    # 計算力量投影
    proxies = _compute_force_proxies(df)
    df[["HL_ret", "pseudo_delta"]] = proxies

    y_all = proxies[["HL_ret", "pseudo_delta"]].values  # shape (T, 2)
    T = y_all.shape[0]

    # --- 狀態空間矩陣 ---
    F = np.array(
        [
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )  # (3,3)

    # 兩個觀測都看 Level
    H = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )  # (2,3)

    # 預設 process noise Q、observation noise R
    if Q is None:
        Q = np.diag([1e-5, 1e-4, 1e-3]).astype(float)  # (3,3)

    if R is None:
        # 兩個觀測通道各自的噪音
        R = np.diag([1e-1, 1e-1]).astype(float)  # (2,2)

    # 初始狀態
    if init_state is None:
        init_x = np.zeros(3)  # [L,S,A]
    else:
        init_x = np.asarray(init_state, dtype=float)

    init_P = np.eye(3) * 1.0  # 初始協方差

    # 儲存 rolling window 之後的狀態估計
    level = np.full(T, np.nan)
    slope = np.full(T, np.nan)
    accel = np.full(T, np.nan)

    I = np.eye(3)

    for t in range(T):
        # 視窗起點
        start = max(0, t - window + 1)
        y = y_all[start : t + 1]  # shape (L, 2)，L<=window
        L = y.shape[0]

        # 每一段視窗都從同一組初始狀態開始（忘記遠古歷史）
        x = init_x.copy()
        P = init_P.copy()

        for i in range(L):
            # 1) 預測
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # 2) 更新
            y_t = y[i]
            y_pred = H @ x_pred
            S = H @ P_pred @ H.T + R  # (2,2)
            K = P_pred @ H.T @ np.linalg.inv(S)  # (3,2)
            innov = y_t - y_pred

            x = x_pred + K @ innov
            P = (I @ P_pred) - K @ H @ P_pred

        # 視窗最後一個點的狀態，填回整體時間軸位置 t
        level[t], slope[t], accel[t] = x[0], x[1], x[2]

    df["kf_level"] = level
    df["kf_slope"] = slope
    df["kf_accel"] = accel

    return df


def compute_rolling_zscore(series, window=50):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def compute_TrendProb(df, window=50, alpha=1.0, beta=1.0):
    df = df.copy()
    df["kc_slope_z"] = compute_rolling_zscore(df["kc_slope"], window)
    df["force_slope_z"] = compute_rolling_zscore(df["kf_slope"], window)
    df["TrendProb"] = sigmoid(alpha * df["kc_slope_z"] + beta * df["force_slope_z"])
    return df


# ---------------------------------------------------
# LSA BuySignal：由 L/S/A 产生最纯数学买点
# ---------------------------------------------------
def LSA_BuySignal_kf(df, k=0.5, std_window=50):
    df = df.copy()

    # Acceleration Cross
    kf_accel_cross = (df["kf_accel"] > 0) & (df["kf_accel"].shift(1) <= 0)

    # Slope & Level Conditions
    kf_slope_ok = df["kf_slope"] > 0
    kf_level_ok = df["kf_level"] > 0

    # 回歸趨勢主幹
    force = df["pseudo_delta"]
    roll_std_force = force.rolling(std_window).std().bfill()
    kf_back_to_level = (force - df["kf_level"]).abs() < k * roll_std_force

    # TrendProb 機率融合
    df = compute_TrendProb(df, window=300)

    # 最終 Buy Signal 條件
    df["BuySignal"] = (
        (kf_accel_cross & kf_slope_ok & kf_level_ok & kf_back_to_level).astype(int)
        # & (df["TrendProb"] > 0.55)
    ).astype(int)

    return df


def LSA_BuySignal_kc(df, k=0.5, std_window=50):
    df = df.copy()

    # Acceleration Cross
    kc_accel_cross = (df["kc_accel"] > 0) & (df["kc_accel"].shift(1) <= 0)

    # Slope & Level Conditions
    kc_slope_ok = df["kc_slope"] > 0
    kc_level_ok = df["kc_level"] > 0

    # 回歸趨勢主幹
    force = df["Close"]
    roll_std_force = force.rolling(std_window).std().bfill()
    kc_back_to_level = (force - df["kc_level"]).abs() < k * roll_std_force

    # TrendProb 機率融合
    df = compute_TrendProb(df, window=300)

    # 最終 Buy Signal 條件
    df["BuySignal"] = (
        (kc_accel_cross & kc_slope_ok & kc_level_ok & kc_back_to_level).astype(int)
        # & (df["TrendProb"] > 0.55)
    ).astype(int)

    return df


def LSA_BuySignal_kc_trendprod(df, k=0.5, std_window=50):
    df = df.copy()

    # Acceleration Cross
    kc_accel_cross = (df["kc_accel"] > 0) & (df["kc_accel"].shift(1) <= 0)

    # Slope & Level Conditions
    kc_slope_ok = df["kc_slope"] > 0
    kc_level_ok = df["kc_level"] > 0

    # 回歸趨勢主幹
    force = df["Close"]
    roll_std_force = force.rolling(std_window).std().bfill()
    kc_back_to_level = (force - df["kc_level"]).abs() < k * roll_std_force

    # TrendProb 機率融合
    df = compute_TrendProb(df, window=300)

    # 最終 Buy Signal 條件
    df["BuySignal"] = (
        (kc_accel_cross & kc_slope_ok & kc_level_ok & kc_back_to_level).astype(int)
        & (df["TrendProb"] > 0.55)
    ).astype(int)

    return df


# ---------------------------------------------------
# LSA BuySignal：由 L/S/A 产生最纯数学买点
# ---------------------------------------------------
def LSA_BuySignal_kf_trendprod(df, k=0.5, std_window=50):
    df = df.copy()

    # Acceleration Cross
    kf_accel_cross = (df["kf_accel"] > 0) & (df["kf_accel"].shift(1) <= 0)

    # Slope & Level Conditions
    kf_slope_ok = df["kf_slope"] > 0
    kf_level_ok = df["kf_level"] > 0

    # 回歸趨勢主幹
    force = df["pseudo_delta"]
    roll_std_force = force.rolling(std_window).std().bfill()
    kf_back_to_level = (force - df["kf_level"]).abs() < k * roll_std_force

    # TrendProb 機率融合
    df = compute_TrendProb(df, window=300)

    # 最終 Buy Signal 條件
    df["BuySignal"] = (
        (kf_accel_cross & kf_slope_ok & kf_level_ok & kf_back_to_level).astype(int)
        & (df["TrendProb"] > 0.55)
    ).astype(int)

    return df


def LSA_BuySignal_kfkc(df, k=0.5, std_window=50):
    df = df.copy()

    # Acceleration Cross
    kf_accel_cross = (df["kf_accel"] > 0) & (df["kf_accel"].shift(1) <= 0)
    kc_accel_cross = (df["kc_accel"] > 0) & (df["kc_accel"].shift(1) <= 0)

    # Slope & Level Conditions
    kf_slope_ok = df["kf_slope"] > 0
    kf_level_ok = df["kf_level"] > 0
    kc_slope_ok = df["kc_slope"] > 0
    kc_level_ok = df["kc_level"] > 0

    # 回歸趨勢主幹
    force = df["pseudo_delta"]
    close = df["Close"]
    roll_std_force = force.rolling(std_window).std().bfill()
    roll_std_close = close.rolling(std_window).std().bfill()
    kf_back_to_level = (force - df["kf_level"]).abs() < k * roll_std_force
    kc_back_to_level = (close - df["kc_level"]).abs() < k * roll_std_close

    # TrendProb 機率融合
    df = compute_TrendProb(df, window=300)

    # 最終 Buy Signal 條件
    df["BuySignal"] = (
        (
            kf_accel_cross
            & kf_slope_ok
            & kf_level_ok
            & kc_accel_cross
            & kc_slope_ok
            & kc_level_ok
            & kf_back_to_level
            & kc_back_to_level
        ).astype(int)
        # & (df["TrendProb"] > 0.55)
    ).astype(int)

    return df


def LSA_BuySignal_kfkc_trendprod(df, k=0.5, std_window=50):
    df = df.copy()

    # Acceleration Cross
    kf_accel_cross = (df["kf_accel"] > 0) & (df["kf_accel"].shift(1) <= 0)
    kc_accel_cross = (df["kc_accel"] > 0) & (df["kc_accel"].shift(1) <= 0)

    # Slope & Level Conditions
    kf_slope_ok = df["kf_slope"] > 0
    kf_level_ok = df["kf_level"] > 0
    kc_slope_ok = df["kc_slope"] > 0
    kc_level_ok = df["kc_level"] > 0

    # 回歸趨勢主幹
    force = df["pseudo_delta"]
    close = df["Close"]
    roll_std_force = force.rolling(std_window).std().bfill()
    roll_std_close = close.rolling(std_window).std().bfill()
    kf_back_to_level = (force - df["kf_level"]).abs() < k * roll_std_force
    kc_back_to_level = (close - df["kc_level"]).abs() < k * roll_std_close

    # TrendProb 機率融合
    df = compute_TrendProb(df, window=300)

    # 最終 Buy Signal 條件
    df["BuySignal"] = (
        (
            kf_accel_cross
            & kf_slope_ok
            & kf_level_ok
            & kc_accel_cross
            & kc_slope_ok
            & kc_level_ok
            & kf_back_to_level
            & kc_back_to_level
        ).astype(int)
        & (df["TrendProb"] > 0.55)
    ).astype(int)

    return df


def LSA_BuySignal_TF3(
    df,
    k=0.5,
    std_window=50,
    comp_window=20,
    exp_thresh=1,
    use_col="HL_ret",
    eps=1e-9,
):
    """
    啟動條件版 LSA BuySignal：
    條件 = A 由負轉正 + S > 0 + proxy 回到 level 骨架 + TF3 壓縮後首次擴張

    df: 已包含 ['Open','High','Low','Close','Vol','level','slope','accel', use_col]
    k:   回到骨架的容忍倍數
    std_window: 計算 proxy 標準差的 rolling 視窗
    comp_window: 壓縮期視窗長度（看過去幾根都沒擴張）
    exp_thresh: range_expansion 大於多少視為「擴張」
    use_col: 用哪個投影來量測「是否回到骨架」，預設 'HL_ret'
    """

    df = df.copy()

    df["range_expansion"] = (df.High - df.Low) / (df.ATR + eps)
    # ------------------------------
    # 1) L / S / A 條件
    # ------------------------------
    # 1. 加速由負轉正
    accel_cross = (df["kf_accel"] > 0) & (df["kf_accel"].shift(1) <= 0)

    # 2. 趨勢方向為正
    slope_ok = df["kf_slope"] > 0

    # 3. 力量投影回到骨架：|proxy - level| < k * std(proxy)
    proxy = df[use_col]
    roll_std_proxy = proxy.rolling(std_window).std().bfill()
    back_to_level = (proxy - df["kf_level"]).abs() < k * roll_std_proxy

    # ------------------------------
    # 2) TF3：壓縮 → 首次擴張
    # ------------------------------
    # 壓縮期：過去 comp_window 根內 range_expansion 都沒有超過 exp_thresh
    # （用前一根為止的歷史判斷，避免偷看未來）
    past_max = df["range_expansion"].rolling(comp_window).max().shift(1)
    was_compressed = past_max < exp_thresh  # 之前都沒爆

    # 擴張點：這一根 range_expansion 超過門檻
    is_expansion = df["range_expansion"] > exp_thresh

    # tf3_trigger = was_compressed & is_expansion
    tf3_trigger = is_expansion

    # ------------------------------
    # 3) 組合成 BuySignal
    # ------------------------------
    df["BuySignal"] = (accel_cross & slope_ok & back_to_level & tf3_trigger).astype(int)

    return df


def add_addon_signal(df, atr_window=14, atr_factor=1.2):
    """
    df 必须已包含: ['close','level','slope','accel']
    输出: 新增一栏 'addon' (0/1)
    """

    out = df.copy()

    # --- 计算 ATR ---
    high = out["High"]
    low = out["Low"]
    close = out["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    out["ATR"] = tr.rolling(atr_window).mean()

    # --- 趋势强化条件（Trend Reinforcement） ---
    cond_slope_up = out["kf_slope"] > out["kf_slope"].shift(1)
    cond_accel_pos = out["kf_accel"] > 0
    cond_atr_ok = out["ATR"] < out["ATR"].shift(1) * atr_factor
    cond_trend_pos = out["kf_slope"] > 0  # 确认方向

    out["addon"] = (
        cond_slope_up & cond_accel_pos & cond_atr_ok & cond_trend_pos
    ).astype(int)

    return out


# 實際跑一次
if __name__ == "__main__":
    file_name = "/Users/zen/code/bayes_excersice/reports/1216_044620_btcusdt_1min_fun1000.0cap100.0atr60bw20up60lw60hmm3_cut0.975pnl3.0ext1.0stp5.csv"
    df = pd.read_csv(f"{file_name}")
    out = kalman_trend_force_multi_obs(df, window=240)
    out = kalman_close_3rd(out, window=240, qL=1e-5, qS=1e-4, qA=1e-3, r0=1e-1)
    out_kf = LSA_BuySignal_kf(out)
    out_kfkc = LSA_BuySignal_kfkc(out)
    out_kf_trendprod = LSA_BuySignal_kf_trendprod(out)
    out_kfkc_trendprod = LSA_BuySignal_kfkc_trendprod(out)
    out_kc = LSA_BuySignal_kc(out)
    out_kc_trendprod = LSA_BuySignal_kc_trendprod(out)
    # out = LSA_BuySignal_TF3(out)

    # out = add_addon_signal(out)
    # print(
    #     out[
    #         [
    #             "Close",
    #             "HL_ret",
    #             "pseudo_delta",
    #             "kf_level",
    #             "kf_slope",
    #             "kf_accel",
    #             "kc_level",
    #             "kc_slope",
    #             "kc_accel",
    #             "TrendProb",
    #         ]
    #     ].head(10)
    # )
    out.to_csv("temp.csv", index=False)
    print("temp.csv")

    # =========================
    # 收集各策略的 profit 样本
    # =========================

    strategy_dict = {}

    st_base = df.profit.dropna()

    st_HMM = df[df.HMM_Signal == 1].profit.dropna()
    st_EMA_Buy = df[(df.BuySignal == "True") | (df.BuySignal == "1.0")].profit.dropna()
    st_HMM_EMA_Buy = df[
        ((df.BuySignal == "True") | (df.BuySignal == "1.0")) & (df.HMM_Signal == 1)
    ].profit.dropna()

    st_kf = out_kf[out_kf.BuySignal == 1].profit.dropna()
    st_kc = out_kc[out_kc.BuySignal == 1].profit.dropna()
    st_kf_kc = out_kfkc[out_kfkc.BuySignal == 1].profit.dropna()

    st_kf_trendprod = out_kf_trendprod[out_kf_trendprod.BuySignal == 1].profit.dropna()
    st_kc_trendprod = out_kc_trendprod[out_kc_trendprod.BuySignal == 1].profit.dropna()
    st_kf_kc_trendprod = out_kfkc_trendprod[
        out_kfkc_trendprod.BuySignal == 1
    ].profit.dropna()

    # =========================
    # 样本量检查函数
    # =========================

    def checksample(strategy_name, profits):
        result = check_sample_size(profits)
        print(f"{strategy_name}: {result['status']}")
        return result["enough"]

    # =========================
    # 统一样本量（KDE + MC）
    # 目标：与 BASE 交易次数一致
    # =========================

    base_n = st_base.count()

    if checksample("BASE", st_base):
        strategy_dict["base"] = st_base

    if checksample("HMM", st_HMM):
        strategy_dict["HMM"] = simulate_profit_by_kde(st_HMM, base_n)

    if checksample("EMA", st_EMA_Buy):
        strategy_dict["EMA_Buy"] = simulate_profit_by_kde(st_EMA_Buy, base_n)

    if checksample("HMM+EMA", st_HMM_EMA_Buy):
        strategy_dict["HMM+EMA_Buy"] = simulate_profit_by_kde(st_HMM_EMA_Buy, base_n)

    if checksample("KF", st_kf):
        strategy_dict["kf"] = simulate_profit_by_kde(st_kf, base_n)

    if checksample("KC", st_kc):
        strategy_dict["kc"] = simulate_profit_by_kde(st_kc, base_n)

    if checksample("KF+KC", st_kf_kc):
        strategy_dict["kf+kc"] = simulate_profit_by_kde(st_kf_kc, base_n)

    if checksample("KF+Trendprod", st_kf_trendprod):
        strategy_dict["kf+trendprod"] = simulate_profit_by_kde(st_kf_trendprod, base_n)

    if checksample("KC+Trendprod", st_kc_trendprod):
        strategy_dict["kc+trendprod"] = simulate_profit_by_kde(st_kc_trendprod, base_n)

    if checksample("KF+KC+Trendprod", st_kf_kc_trendprod):
        strategy_dict["kf+kc+trendprod"] = simulate_profit_by_kde(
            st_kf_kc_trendprod, base_n
        )

    # =========================
    # 输出中间结果（原始 dataframe）
    # =========================

    out_kf.to_csv(
        f"{file_name}_KF.csv",
        index=False,
    )
    out_kc.to_csv(
        f"{file_name}_KC.csv",
        index=False,
    )
    out_kfkc.to_csv(
        f"{file_name}_KF_KC.csv",
        index=False,
    )
    out_kf_trendprod.to_csv(
        f"{file_name}_KF_TrendProd.csv",
        index=False,
    )
    out_kc_trendprod.to_csv(
        f"{file_name}_KC_TrendProd.csv",
        index=False,
    )
    out_kfkc_trendprod.to_csv(
        f"{file_name}_KF_KC_TrendProd.csv",
        index=False,
    )

    # =========================
    # 各策略 KDE / 分布比较
    # =========================

    plot_strategy_kde_comparison(strategy_dict)
    plot_strategy_cdf_comparison(strategy_dict)
    plot_kl_matrix(strategy_dict)
    plot_tail_quadrant(strategy_dict)
