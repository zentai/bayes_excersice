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


if __name__ == "__main__":
    # btc_df = pd.read_csv(
    #     f"/Users/zen/Documents/code/bayes/reports/0101_083626_btcusdt_1day_fun15.0cap10.1atr60bw20up60lw60hmm4_cut0.975pnl3.0ext1.2stp5.csv"
    # )
    # eval_noise_slope_predictive_power_profit(btc_df)

    # baba_df = pd.read_csv(f"/Users/zen/Documents/code/bayes//reports/BABA_cached.csv")
    # eval_noise_slope_predictive_power_profit(baba_df)

    xrp_df = pd.read_csv(
        f"/Users/zen/Documents/code/bayes/reports/1227_041038_xrpusdt_1day_fun15.0cap10.1atr60bw20up60lw60hmm4_cut0.975pnl3.0ext1.2stp5.csv"
    )
    eval_noise_slope_predictive_power_profit(xrp_df)
