import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_force_vs_price(df):

    # ----------------- m_force -----------------
    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(df["Date"], df["Close"], color="blue", linewidth=1, label="Close")
    ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["m_force"], color="orange", linewidth=0.8, label="m_force")
    ax2.set_ylabel("m_force (std)")
    plt.title("Close vs m_force")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # ----------------- m_force_bias -----------------
    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(df["Date"], df["Close"], color="blue", linewidth=1, label="Close")
    ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    ax2.plot(
        df["Date"],
        df["m_force_bias"],
        color="orange",
        linewidth=0.8,
        label="m_force_bias",
    )
    ax2.set_ylabel("m_force_bias (std)")
    plt.title("Close vs m_force_bias")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # ----------------- regime_noise_level -----------------
    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(df["Date"], df["Close"], color="blue", linewidth=1, label="Close")
    ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    ax2.plot(
        df["Date"],
        df["m_regime_noise_level"],
        color="green",
        linewidth=1,
        label="m_regime_noise_level",
    )
    ax2.set_ylabel("m_regime_noise_level")
    plt.title("Close vs regime_noise_level")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def backtest_force_efficiency(df, force_col="m_force"):
    """
    验证：力量越大，策略获利是否越佳？
    """
    valid = df.dropna(subset=[force_col, "profit"])

    # 以力量作分箱
    valid["force_bucket"] = pd.qcut(valid[force_col], q=10, duplicates="drop")

    summary = (
        valid.groupby("force_bucket")["profit"]
        .agg(["count", "mean", "median", "sum"])
        .sort_index()
    )

    print("\n=== Force Efficiency Test ===")
    print(summary)

    # 简单plot：力量 vs 平均报酬
    plt.figure(figsize=(12, 5))
    plt.bar(summary.index.astype(str), summary["mean"])
    plt.title(f"Force vs Mean Profit ({force_col})")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

    return summary


def main():
    # TODO: 修改成你回测 CSV 的路径
    csv_path = "/Users/Zen/Documents/code/bayes_excersice/reports/1230_115024_btcusdt_1day_fun15.0cap15.0atr60bw20up60lw60hmm4_cut0.975pnl3.0ext1.2stp5.csv"
    df = pd.read_csv(csv_path)

    # 若 Date 是秒或毫秒，需做转换
    if np.issubdtype(df["Date"].dtype, np.number):
        df["Date"] = pd.to_datetime(df["Date"], unit="ms", errors="ignore")

    plot_force_vs_price(df)

    print("\n**Efficiency check for m_force**")
    backtest_force_efficiency(df, "m_force")

    print("\n**Efficiency check for m_force_bias**")
    backtest_force_efficiency(df, "m_force_bias")


if __name__ == "__main__":
    main()
