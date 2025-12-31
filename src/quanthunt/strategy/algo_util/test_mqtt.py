# main.py
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from quanthunt.strategy.algo_util.mqtt_ui import publish_sync, publish_update


def make_order(n, close):
    o = [None] * n
    for i in range(20, n, 40):
        o[i] = {"side": "BUY", "price": float(close[i])}
    for i in range(35, n, 50):
        o[i] = {"side": "SELL", "price": float(close[i])}
    return o


def generate_continuous_df(n=250):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    dates = [now - timedelta(hours=(n - i)) for i in range(n)]
    close = 100 + np.cumsum(np.random.normal(0, 0.4, n))
    m_force = np.random.normal(0, 1.0, n)

    # regime 隨機長度塊
    regime = []
    cur = 0
    cnt = 0
    for _ in range(n):
        if cnt == 0:
            cur = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            cnt = np.random.randint(10, 25)
        regime.append(cur)
        cnt -= 1

    regime = np.array(regime)
    hmm_signal = np.where(regime == 1, np.random.binomial(1, 0.4, n), 0)
    cp_prob = np.random.uniform(0, 0.2, n)
    risk = np.random.uniform(0.1, 0.9, n)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "m_force": m_force,
            "regime": regime,
            "hmm_signal": hmm_signal,
            "cp_prob": cp_prob,
            "risk": risk,
            "order": make_order(n, close),
        }
    )

    df.set_index("Date", inplace=True)
    return df


if __name__ == "__main__":
    symbols = ["BTC", "XRP", "ADA"]
    interval = "1hr"
    print("⏳ 測試資料開始 ...")

    for sym in symbols:
        df = generate_continuous_df(250)

        batch_df = df.iloc[:200]
        live_df = df.iloc[200:]

        print(f"📤 publish_sync({sym}) → 200 rows")
        publish_sync(sym, interval, batch_df)
        time.sleep(1.0)

        print(f"🚀 publish_update({sym}) → 50 rows")
        for idx, row in live_df.iterrows():
            publish_update(sym, interval, row)
            print(f"[{idx}] Close={row['Close']:.2f}")
            time.sleep(1.0)

    print("🎉 All done.")
