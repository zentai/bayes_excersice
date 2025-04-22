import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from config import config

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir


def reduce_noise_by_wavelet_transform(df):
    # 取HIGH作为价格
    prices = df["High"].values

    # 使用Daubechies小波（db4）进行三层分解
    coeffs = pywt.wavedec(prices, "db4", level=3)

    # 软阈值处理
    threshold = 0.5

    def soft_thresholding(coeffs, threshold):
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)

    coeffs[1:] = [soft_thresholding(c, threshold) for c in coeffs[1:]]

    # 重构去噪后的信号
    denoised_prices = pywt.waverec(coeffs, "db4")

    # 将去噪后的价格放入新列
    df["price"] = denoised_prices[: len(df)]

    return df


### 2. `calculate_acc_speed` 计算价格和成交量的一阶导数（速度）和二阶导数（加速度）。
def calculate_acc_speed(df, moving_window=5):
    # 计算价格的一阶导数（速度）
    df["price_speed"] = df["price"].diff()

    # 计算价格的二阶导数（加速度）
    df["price_acc"] = df["price_speed"].diff()

    # 计算成交量的一阶导数（速度）
    df["vol_speed"] = df["Vol"].diff()

    # 计算成交量的二阶导数（加速度）
    df["vol_acc"] = df["vol_speed"].diff()

    return df


### 3. `chartting` 绘制价格、成交量、价格加速度和成交量加速度的图表。
def chartting(df):
    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(df["Date"], df["price"])
    plt.title("Price")

    plt.subplot(4, 1, 2)
    plt.plot(df["Date"], df["Vol"])
    plt.title("Vol")

    plt.subplot(4, 1, 3)
    plt.plot(df["Date"], df["price_acc"])
    plt.title("Price Acceleration")

    plt.subplot(4, 1, 4)
    plt.plot(df["Date"], df["vol_acc"])
    plt.title("Vol Acceleration")

    plt.tight_layout()
    plt.show()


### 4. `statistics` 统计各种成交量和价格加速度的组合分布。


def statistics(df):
    vection_dict = {}

    for i in range(len(df) - 5):
        price_acc = df.loc[i, "price_acc"]
        vol_acc = df.loc[i, "vol_acc"]

        key = (price_acc, vol_acc)
        if key not in vection_dict:
            vection_dict[key] = 1
        else:
            vection_dict[key] += 1

    return vection_dict


### 5. `chart_statistic` 绘制统计分布图。
def chart_statistic(vection_dict):
    keys = list(vection_dict.keys())
    values = list(vection_dict.values())

    price_acc = [k[0] for k in keys]
    vol_acc = [k[1] for k in keys]

    plt.figure(figsize=(10, 8))
    plt.scatter(price_acc, vol_acc, s=values, alpha=0.5)
    plt.title("Price Acceleration vs Vol Acceleration Distribution")
    plt.xlabel("Price Acceleration")
    plt.ylabel("Vol Acceleration")
    plt.show()


def debug_wavelet_transform(df, column="High", wavelet="db4", level=3, threshold=0.5):
    # 取HIGH作为价格
    original_prices = df[column].values

    # 使用指定小波进行分解
    coeffs = pywt.wavedec(original_prices, wavelet, level=level)

    # 软阈值处理
    def soft_thresholding(coeffs, threshold):
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)

    coeffs[1:] = [soft_thresholding(c, threshold) for c in coeffs[1:]]

    # 重构去噪后的信号
    denoised_prices = pywt.waverec(coeffs, wavelet)

    # 将去噪后的价格放入新列
    df["price"] = denoised_prices[: len(df)]

    # 画出小波变换前后的图
    plt.figure(figsize=(14, 7))

    plt.plot(df["Date"], original_prices, label="Original Prices", color="blue")
    plt.plot(
        df["Date"], df["price"], label="Denoised Prices", color="red", linestyle="--"
    )
    plt.title("Original and Denoised Prices (Wavelet Transform)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    plt.tight_layout()
    plt.show()


### 6. `main` 完整的 `main` 函数实现。
def main(df):
    # df = reduce_noise_by_wavelet_transform(df)  # 取HIGH作为价格，将移除噪声之后的价格放入新column price
    df = debug_wavelet_transform(df)
    df = calculate_acc_speed(
        df, moving_window=5
    )  # 使用T与T-5进行计算，这样能确保最新的T一定会有资料，并将价格加速度放入 price_acc, 成交量加速度放入 vol_acc
    chartting(df)  # 将价格/成交量/价格加速度/成交量加速度 画出来
    vection_dict = statistics(df)  # 统计各种成交量/价格的加速度的组合分布，
    chart_statistic(vection_dict)  # 将统计分布画出来
    calculate_and_plot_correlation(df)


def calculate_and_plot_correlation(df):
    # 计算一阶和二阶导数（速度和加速度）
    df["price_speed"] = df["price"].diff()
    df["price_acc"] = df["price_speed"].diff()
    df["vol_speed"] = df["Vol"].diff()
    df["vol_acc"] = df["vol_speed"].diff()

    # 去掉空值
    df = df.dropna()

    # 计算价格加速度和成交量加速度的相关系数
    correlation = df[["price_acc", "vol_acc"]].corr()
    print("价格加速度和成交量加速度的相关系数:")
    print(correlation)

    # 筛选正的成交量加速度数据
    positive_vol_acc = df[df["vol_acc"] > 0]

    # 绘制正的成交量加速度和价格加速度的关系图
    plt.scatter(positive_vol_acc["price_acc"], positive_vol_acc["vol_acc"], alpha=0.5)
    plt.title("Positive Volume Acceleration vs Price Acceleration")
    plt.xlabel("Price Acceleration")
    plt.ylabel("Volume Acceleration")
    plt.show()

    return correlation


code = "tonusdt"
df = pd.read_csv(f"{DATA_DIR}/{code}_cached.csv")

# 调用主函数
main(df)
