# main.py
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quanthunt.strategy.algo_util.mqtt_ui import publish_sync, publish_update


def make_fake_df(n: int = 200) -> pd.DataFrame:
    """
    產生一份假 1hr K 線資料：
        Close, m_force, regime, hmm_signal, cp_prob, risk, order
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    dates = [now - timedelta(hours=n - i) for i in range(n)]

    close = 100 + np.cumsum(np.random.normal(0, 0.5, n))
    m_force = np.random.normal(0, 1.0, n)

    # regime: -1 / 0 / 1 隨機塊狀
    regime = []
    current = 0
    block = 0
    for i in range(n):
        if block == 0:
            current = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            block = np.random.randint(5, 20)
        regime.append(current)
        block -= 1

    regime = np.array(regime)

    # hmm_signal：只在 regime==1 的部分隨機開啟 0/1
    hmm_signal = np.where(
        regime == 1,
        np.random.binomial(1, 0.6, n),
        0,
    )

    # cp_prob：在 regime 轉折附近拉高
    cp_prob = np.zeros(n)
    for i in range(1, n):
        if regime[i] != regime[i - 1]:
            cp_prob[i] = np.random.uniform(0.6, 0.9)
        else:
            cp_prob[i] = max(cp_prob[i - 1] * 0.8, np.random.uniform(0.0, 0.2))

    # risk：隨機 + regime / m_force 粗略組合
    risk = np.clip(
        0.2
        + 0.2 * (regime < 0)
        + 0.1 * (np.abs(m_force) > 1.5)
        + np.random.normal(0, 0.05, n),
        0.0,
        1.0,
    )

    df = pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "m_force": m_force,
            "regime": regime,
            "hmm_signal": hmm_signal,
            "cp_prob": cp_prob,
            "risk": risk,
        }
    ).set_index("Date")

    # 加一個 order 欄位，某些點標記 BUY / SELL
    orders = [None] * n
    for i in range(10, n, 40):
        orders[i] = {"side": "BUY", "price": float(close[i])}
    for i in range(30, n, 50):
        orders[i] = {"side": "SELL", "price": float(close[i])}
    df["order"] = orders

    return df


if __name__ == "__main__":
    symbols = ["BTC", "XRP", "ADA"]
    interval = "1hr"

    print("⏳ 生成假資料並透過 MQTT 測試 Dashboard ...")

    # 先對每個 symbol 做一次 batch sync
    for sym in symbols:
        df = make_fake_df(200)
        print(f"📤 publish_sync({sym}, {interval}) with {len(df)} rows")
        publish_sync(sym, interval, df)
        time.sleep(0.5)

    # 然後對 BTC 做即時更新模擬
    sym = "BTC"
    df_live = make_fake_df(50)
    print(f"🚀 模擬 {sym} 即時更新 50 筆 ...")
    for idx, row in df_live.iterrows():
        publish_update(sym, interval, row)
        print(
            f"[{idx}] Close={row['Close']:.2f}, m_force={row['m_force']:.2f}, "
            f"regime={row['regime']}, hmm={row['hmm_signal']}, risk={row['risk']:.2f}"
        )
        time.sleep(1.0)

    print("🎉 測試完成，請在瀏覽器查看 Dashboard。")
