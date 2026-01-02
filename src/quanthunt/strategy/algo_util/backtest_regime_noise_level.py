import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _rolling_slope(y: np.ndarray) -> float:
    """
    對長度 = window 的一段 y，計算 y = a + b * x 的 b（slope）
    x 用 0,1,2,...,window-1
    """
    x = np.arange(len(y), dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    if den == 0:
        return 0.0
    return num / den


def eval_noise_slope_predictive_power_profit(df: pd.DataFrame):
    """
    df 必須包含欄位：
        - profit  （已結束交易的真實報酬）
        - m_regime_noise_level

    回傳：
        - 噪音 slope vs profit 分桶效果圖
        - WinRate / MeanProfit 表格
    """

    tmp = df.copy().reset_index(drop=True)

    # === 噪音斜率 ===
    WINDOW = 60
    tmp["noise"] = tmp["m_regime_noise_level"]
    tmp["noise_slope"] = (
        tmp["noise"]
        .rolling(WINDOW, min_periods=WINDOW)
        .apply(_rolling_slope, raw=True)
        .shift(1)  # 避免看當日完整價量後才下定義
    )

    # 只取已成熟交易的資料（profit != NaN）
    tmp = tmp.dropna(subset=["profit", "noise_slope"]).copy()

    # === 依斜率分桶（五等分）===
    qs = tmp["noise_slope"].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()

    def bucket_noise(x):
        if x <= qs[0.2]:
            return "Q1-噪音大幅下降"
        if x <= qs[0.4]:
            return "Q2"
        if x <= qs[0.6]:
            return "Q3"
        if x <= qs[0.8]:
            return "Q4"
        return "Q5-噪音大幅上升"

    tmp["noise_bucket"] = tmp["noise_slope"].map(bucket_noise)

    # === 效果測量 ===
    summary = (
        tmp.groupby("noise_bucket")["profit"]
        .agg(
            count="size",
            win_rate=lambda s: (s > 0).mean(),
            mean_profit="mean",
            median_profit="median",
            p90=lambda s: s.quantile(0.9),
            p10=lambda s: s.quantile(0.1),
        )
        .reset_index()
    )

    print("\n=== 噪音 slope 分桶 → Profit 結果統計 ===")
    print(summary)

    # === Plot 圖 ===
    bucket_order = ["Q1-噪音大幅下降", "Q2", "Q3", "Q4", "Q5-噪音大幅上升"]
    data_to_plot = [tmp[tmp["noise_bucket"] == b]["profit"] for b in bucket_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        data_to_plot,
        labels=bucket_order,
        showmeans=True,
        meanline=True,
        whis=1.5,
    )
    ax.set_title("Noise Slope vs Profit (Actual P/L)")
    ax.set_ylabel("Profit / P&L")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    return summary


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_liquidity_hunt_analysis(
    df: pd.DataFrame,
    z_window: int = 200,
    force_z_th: float = 0.25,
    noise_th: float = 1.05,
    cp_slope_th: float = 0.0,
    horizon: int = 30,
):
    """
    v3: 事件條件改為：
      force_z > force_z_th
      & m_regime_noise_level < noise_th
      & cp_diff > cp_slope_th
    """

    df = df.copy().reset_index(drop=True)

    # --- 1) 準備 z-score & cp_diff ---
    roll_mean = df["m_force"].rolling(z_window).mean()
    roll_std = df["m_force"].rolling(z_window).std().replace(0, np.nan)

    df["force_z"] = ((df["m_force"] - roll_mean) / roll_std).fillna(0.0)
    df["cp_diff"] = df["bocpd_cp_prob"].diff().fillna(0.0)

    # --- 2) 事件判定 ---
    df["HuntEvent"] = (
        (df["force_z"] > force_z_th)
        & (df["m_regime_noise_level"] < noise_th)
        & (df["cp_diff"] > cp_slope_th)
    )

    events = df.index[df["HuntEvent"]].tolist()

    fig, axes = plt.subplots(3, 2, figsize=(20, 14))
    axes = axes.flatten()

    def mark_events(ax):
        for e in events:
            ax.axvline(e, color="red", alpha=0.2, linewidth=0.8)

    # 1) Noise vs Price
    ax = axes[0]
    ax.plot(df["Close"], color="tab:blue", label="Close")
    ax2 = ax.twinx()
    ax2.plot(df["m_regime_noise_level"], color="tab:green", alpha=0.6, label="Noise")
    ax.set_title("Noise vs Price")
    mark_events(ax)
    ax.grid(True)

    # 2) BOCPD cp_prob & cp_diff
    ax = axes[1]
    ax.plot(df["Close"], color="tab:blue", label="Close")
    ax2 = ax.twinx()
    ax2.plot(df["bocpd_cp_prob"], color="tab:orange", alpha=0.7, label="cp_prob")
    ax2.plot(df["cp_diff"], color="tab:red", alpha=0.4, label="cp_diff")
    ax.set_title("BOCPD cp_prob & cp_diff")
    mark_events(ax)
    ax.grid(True)

    # 3) Volume vs Price
    ax = axes[2]
    ax.plot(df["Close"], color="tab:blue")
    ax2 = ax.twinx()
    ax2.fill_between(df.index, df["Vol"], alpha=0.3, color="tab:green")
    ax.set_title("Volume vs Price")
    mark_events(ax)
    ax.grid(True)

    # 4) Force & force_z vs Price
    ax = axes[3]
    ax.plot(df["Close"], color="tab:blue")
    ax2 = ax.twinx()
    ax2.fill_between(
        df.index, df["m_force"], alpha=0.3, color="tab:pink", label="Force"
    )
    ax2.plot(df["force_z"], color="tab:red", alpha=0.6, label="force_z")
    ax2.axhline(force_z_th, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Force & force_z vs Price")
    mark_events(ax)
    ax.grid(True)

    # 5) Profit vs Price
    ax = axes[4]
    ax.plot(df["Close"], color="tab:blue")
    ax2 = ax.twinx()
    ax2.fill_between(df.index, df["profit"], alpha=0.3, color="tab:purple")
    ax.set_title("Profit movement")
    mark_events(ax)
    ax.grid(True)

    # 6) Event return histogram
    event_returns = []
    for e in events:
        if e + horizon < len(df):
            r = df["Close"].iloc[e + horizon] - df["Close"].iloc[e]
            event_returns.append(r)

    ax = axes[5]
    if len(event_returns) > 0:
        ax.hist(event_returns, bins=20, alpha=0.7, color="tab:cyan")
    ax.set_title(f"Return {horizon} bars after event")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"事件數量: {len(events)}")
    if len(event_returns) > 0:
        er = np.array(event_returns)
        print(f"平均報酬: {er.mean():.6f}")
        print(f"中位數報酬: {np.median(er):.6f}")
        print(f"勝率: {(er > 0).mean():.2%}")


def build_liquidity_short_filter(
    df: pd.DataFrame,
    *,
    force_abs_th: float = 0.5,
    noise_th: float = 1.05,
    cp_diff_th: float = 0.0,
):
    """
    Liquidity-follow short filter

    直覺條件：
    1. 世界不允許做多（HMM_Signal == 0）
    2. 多方已撤退（m_z_force < 0）
    3. 沒有人在用力砸（|m_z_force| 很小）
    4. 流動性變薄（noise 低）
    5. 結構正在鬆（BOCPD 變盤機率上升）
    """

    df = df.copy()

    # BOCPD 斜率（結構是否開始鬆動）
    df["cp_diff"] = df["bocpd_cp_prob"].diff().fillna(0.0)

    df["short_filter"] = (
        (df["HMM_Signal"] == 0)
        & (df["m_z_force"] < 0)
        & (df["m_z_force"].abs() < force_abs_th)
        & (df["m_regime_noise_level"] < noise_th)
        & (df["cp_diff"] > cp_diff_th)
    )

    return df


import matplotlib.pyplot as plt


def plot_analysis(df: pd.DataFrame):

    df = df.copy()

    # === 多單條件（維持你原本的，用來對照）===
    bull_cond = (df["m_z_force"] > 1.0) & (df["HMM_Signal"] == 1)

    # === 空單條件：來自 liquidity filter ===
    short_cond = df.get("short_filter", False)

    df["direction_signal"] = 0
    df.loc[bull_cond, "direction_signal"] = 1
    df.loc[short_cond, "direction_signal"] = -1

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    # --- Price ---
    ax.plot(df["Close"].values, label="Close", color="steelblue")

    # --- Force ---
    ax2 = ax.twinx()
    ax2.plot(df["m_z_force"].values, color="salmon", alpha=0.35, label="m_z_force")
    ax2.axhline(0, color="gray", alpha=0.3)

    # --- Signals ---
    long_idx = df[df["direction_signal"] == 1].index
    short_idx = df[df["direction_signal"] == -1].index

    ax.scatter(
        long_idx,
        df.loc[long_idx, "Close"],
        color="lime",
        s=30,
        label="Long (Force + HMM)",
        zorder=5,
    )

    ax.scatter(
        short_idx,
        df.loc[short_idx, "Close"],
        color="purple",
        s=30,
        label="Short (Liquidity-follow)",
        zorder=5,
    )

    # --- BOCPD shock marker ---
    shock_dates = df[df["bocpd_cp_prob"].diff().abs() > 0.001].index
    for d in shock_dates:
        ax.axvline(d, color="red", alpha=0.12)

    ax.set_title("Price + Force + Liquidity-follow Short Filter")
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # btc_df = pd.read_csv(
    #     f"/Users/zen/Documents/code/bayes/reports/0101_083626_btcusdt_1day_fun15.0cap10.1atr60bw20up60lw60hmm4_cut0.975pnl3.0ext1.2stp5.csv"
    # )
    # eval_noise_slope_predictive_power_profit(btc_df)

    # baba_df = pd.read_csv(f"/Users/zen/Documents/code/bayes//reports/BABA_cached.csv")
    # eval_noise_slope_predictive_power_profit(baba_df)

    xrp_df = pd.read_csv(
        f"/Users/zen/Documents/code/bayes/reports/0102_094308_xrpusdt_60min_fun15.0cap10.1atr60bw20up60lw60hmm4_cut0.975pnl3.0ext1.2stp5.csv"
    )
    df2 = build_liquidity_short_filter(
        xrp_df,
        force_abs_th=0.5,
        noise_th=1.05,
        cp_diff_th=0.0,
    )

    plot_analysis(df2)
    plot_liquidity_hunt_analysis(xrp_df)
