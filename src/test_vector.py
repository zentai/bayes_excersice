import pandas as pd
import matplotlib.pyplot as plt

from .sensor.market_sensor import HuobiMarketSensor
from .hunterverse.interface import Symbol

sensor = HuobiMarketSensor(symbol=Symbol("nearusdt"), interval="1min")
df = sensor.scan(2000)

# 整理數據
df["Date"] = pd.to_datetime(df["Date"], unit="ms")
df.set_index("Date", inplace=True)

print(df)

# 計算OBV
df["price_diff"] = df["Close"].diff()
df["direction"] = df["price_diff"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
df["Vol"] *= df["direction"]
df["OBV"] = df["Vol"].cumsum()

# 繪製價格和OBV
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(df.index, df["Close"], color="b", label="Close")
ax1.set_ylabel("Price (USD)", color="b")
ax1.tick_params(axis="y", labelcolor="b")

ax2 = ax1.twinx()
ax2.plot(df.index, df["OBV"], color="r", label="OBV")
ax2.set_ylabel("OBV", color="r")
ax2.tick_params(axis="y", labelcolor="r")

plt.title("BTC Price and OBV")
fig.tight_layout()
plt.show()
