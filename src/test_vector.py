import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .sensor.market_sensor import HuobiMarketSensor
from .hunterverse.interface import Symbol

sensor = HuobiMarketSensor(symbol=Symbol("smileyusdt"), interval="1min")
df = sensor.scan(2000)

df["Date"] = pd.to_datetime(df["Date"], unit="ms")
df.set_index("Date", inplace=True)

df["price_diff"] = df["Close"].diff()
df["direction"] = df["price_diff"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
df["Vol"] *= df["direction"]
df["OBV"] = df["Vol"].cumsum()

# Calculate moving average and standard deviation for OBV
df["OBV_MA"] = df["OBV"].rolling(window=60).mean()
df["OBV_std"] = df["OBV"].rolling(window=60).std()

# Calculate upper and lower bounds
df["upper_bound"] = df["OBV_MA"] + (3 * df["OBV_std"])
df["lower_bound"] = df["OBV_MA"] - (3 * df["OBV_std"])

# Calculate relative difference between bounds
df["bound_diff"] = (df["upper_bound"] - df["lower_bound"]) / df["OBV_MA"]

# Identify significant points where OBV crosses the bounds
df["upper_cross"] = (
    (df["OBV"] > df["upper_bound"])
    & (df["OBV"].shift(1) <= df["upper_bound"])
    & (df["bound_diff"] > 0.05)
)
df["lower_cross"] = (
    (df["OBV"] < df["lower_bound"])
    & (df["OBV"].shift(1) >= df["lower_bound"])
    & (df["bound_diff"] > 0.05)
)

fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(df.index, df["Close"], color="b", label="Close")
ax1.set_ylabel("Price (USD)", color="b")
ax1.tick_params(axis="y", labelcolor="b")

ax2 = ax1.twinx()
ax2.plot(df.index, df["OBV"], color="g", label="OBV")
ax2.plot(df.index, df["upper_bound"], color="gray", linestyle="--", label="Upper Bound")
ax2.plot(df.index, df["lower_bound"], color="gray", linestyle="--", label="Lower Bound")
ax2.set_ylabel("OBV", color="r")
ax2.tick_params(axis="y", labelcolor="r")

# Plot markers for significant upper and lower crosses
ax2.scatter(
    df.index[df["upper_cross"]],
    df["OBV"][df["upper_cross"]],
    color="red",
    marker="^",
    label="Break Upper Bound",
    zorder=5,
)
# ax2.scatter(
#     df.index[df["lower_cross"]],
#     df["OBV"][df["lower_cross"]],
#     color="red",
#     marker="v",
#     label="Break Lower Bound",
#     zorder=5,
# )

plt.title("BTC Price and OBV with Bounds")
ax2.legend(loc="upper left")
fig.tight_layout()
plt.show()
