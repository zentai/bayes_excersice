import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import functools
from strategy.bayes import conditional, prob, odd, prob_odd

tparam = {
    "ATR_sample": 116,
    "atr_loss_margin": 2.00000,
    "bayes_windows": 198,
    "lower_sample": 113,
    "upper_sample": 8,
    "max_invest": 100,
    "bbh_window": 20,
    "bbl_window": 20,
    "bbl_grade": 4,
}


def turtle_trading(base_df):
    upper_sample = int(tparam.get("upper_sample", 20))
    lower_sample = int(tparam.get("lower_sample", 10))
    ATR_sample = int(tparam.get("ATR_sample", 20))

    is_scratch = "ATR" not in base_df.columns
    windows = (
        len(base_df)
        if is_scratch
        else np.max([upper_sample, lower_sample, ATR_sample]) + 1
    )

    # performance: only re-calc nessasary part.
    df = base_df.iloc[-windows:].copy()
    idx = df.index if is_scratch else df[np.isnan(df["ATR"])].index
    df = df.assign(turtle_h=df.Close.shift(1).rolling(upper_sample).max())
    df = df.assign(turtle_l=df.Close.shift(1).rolling(lower_sample).min())
    df = df.assign(h_l=df.High - df.Low)
    df = df.assign(c_h=(df.Close.shift(1) - df.High).abs())
    df = df.assign(c_l=(df.Close.shift(1) - df.Low).abs())
    df = df.assign(TR=df[["h_l", "c_h", "c_l"]].max(axis=1))
    df = df.assign(ATR=(df.TR.rolling(ATR_sample).sum() / ATR_sample))

    # copy value to base_df
    base_df.loc[idx, "turtle_h"] = df.loc[idx, "turtle_h"]
    base_df.loc[idx, "turtle_l"] = df.loc[idx, "turtle_l"]
    base_df.loc[idx, "ATR"] = df.loc[idx, "ATR"]
    return base_df


def bollinger_band(df):
    mv = 30

    idx = df.index
    df = df.assign(avg_close=df.Close.shift(1).rolling(mv).mean())
    df = df.assign(std3=df.avg_close - df.Close.shift(1).rolling(mv).std() * 3)
    df = df.assign(std5=df.avg_close - df.Close.shift(1).rolling(mv).std() * 5)
    df = df.assign(std7=df.avg_close - df.Close.shift(1).rolling(mv).std() * 7)
    df = df.assign(std9=df.avg_close - df.Close.shift(1).rolling(mv).std() * 9)

    payment_std3 = np.where(df["Low"] <= df["std3"], 12.5, 0)
    df["btc_std3"] = payment_std3 / df["std3"]

    payment_std5 = np.where(df["Low"] <= df["std5"], 12.5, 0)
    df["btc_std5"] = payment_std5 / df["std5"]

    payment_std7 = np.where(df["Low"] <= df["std7"], 25.0, 0)
    df["btc_std7"] = payment_std7 / df["std7"]

    payment_std9 = np.where(df["Low"] <= df["std9"], 50.0, 0)
    df["btc_std9"] = payment_std9 / df["std9"]

    # 计算总共购买的比特币数量
    df["total_btc"] = df["btc_std3"] + df["btc_std5"] + df["btc_std7"] + df["btc_std9"]

    # 汇总所有支付金额
    df["total_payment"] = payment_std3 + payment_std5 + payment_std7 + payment_std9

    # 计算平均成本
    df["average_cost"] = df["total_payment"] / df["total_btc"]

    # 显示结果
    print(
        df[["Date", "Low", "total_btc", "total_payment", "average_cost"]][
            df.Low < df.std5
        ]
    )
    return df


def run():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    pd.set_option("display.width", 300)

    code = "BTC-USD"
    df = pd.read_csv(f"data/{code}.csv")
    df = df.dropna()

    df = bollinger_band(df)


if __name__ == "__main__":
    run()
