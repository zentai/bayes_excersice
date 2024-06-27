import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_generative_model(df, window=10):
    """
    计算生成模型并计算惊讶值。

    参数：
    df: DataFrame，包含OHLCV数据
    window: int，用于计算的窗口大小

    返回：
    DataFrame，包含计算后的生成模型预测值和惊讶值
    """
    df["Predicted_Price"] = np.nan
    df["Predicted_Vol"] = np.nan
    df["Updated_Price"] = np.nan
    df["Updated_Vol"] = np.nan
    df["Sigma_P"] = np.nan
    df["Sigma_V"] = np.nan
    df["Surprise"] = np.nan

    for t in range(window, len(df)):
        # 预测步骤：使用历史数据计算滚动平均作为预测值
        predicted_price = df["Close"].iloc[t - window : t].mean()
        predicted_vol = df["Vol"].iloc[t - window : t].mean()

        # 计算误差
        price_error = df["Close"].iloc[t] - predicted_price
        vol_error = df["Vol"].iloc[t] - predicted_vol

        # 计算标准差
        sigma_p = df["Close"].iloc[t - window : t].std()
        sigma_v = df["Vol"].iloc[t - window : t].std()

        # 卡尔曼增益
        K_p = sigma_p**2 / (sigma_p**2 + price_error**2)
        K_v = sigma_v**2 / (sigma_v**2 + vol_error**2)

        # 更新步骤
        updated_price = predicted_price + K_p * price_error
        updated_vol = predicted_vol + K_v * vol_error

        # 存储结果
        df.at[df.index[t], "Predicted_Price"] = predicted_price
        df.at[df.index[t], "Predicted_Vol"] = predicted_vol
        df.at[df.index[t], "Updated_Price"] = updated_price
        df.at[df.index[t], "Updated_Vol"] = updated_vol
        df.at[df.index[t], "Sigma_P"] = sigma_p
        df.at[df.index[t], "Sigma_V"] = sigma_v

        # 计算惊讶值
        surprise = (
            (price_error**2 / (2 * sigma_p**2))
            + (vol_error**2 / (2 * sigma_v**2))
            + 0.5 * np.log(2 * np.pi * sigma_p * sigma_v)
        )
        df.at[df.index[t], "Surprise"] = surprise

    return df


def plot_results(df):
    """
    绘制结果图。

    参数：
    df: DataFrame，包含生成模型预测值和惊讶值
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = "tab:blue"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price", color=color)
    ax1.plot(df.index, df["Close"], label="Close Price", color=color)
    ax1.plot(
        df.index,
        df["Updated_Price"],
        label="Updated Price",
        color="tab:cyan",
        linestyle="dashed",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    # ax3 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Surprise", color=color)
    ax2.plot(df.index, df["Surprise"], label="Surprise", color=color)
    # ax3.plot(df.index, df["Vol"], label="Vol", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.title("Price, Updated Price and Surprise")
    plt.show()


from config import config

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir

code = "5148.KL"
df = pd.read_csv(f"{DATA_DIR}/{code}_cached.csv")
# 计算生成模型
df = compute_generative_model(df)

# 绘图
plot_results(df)
