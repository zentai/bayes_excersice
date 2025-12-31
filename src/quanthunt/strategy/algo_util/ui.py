import json
import paho.mqtt.publish as publish

MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC = "market/close_force"


def publish_update(date, close, m_force):
    payload = {
        "Date": date.isoformat(),  # <-- 統一格式
        "Close": float(close),
        "m_force": float(m_force),
    }
    publish.single(TOPIC, json.dumps(payload), hostname=MQTT_HOST, port=MQTT_PORT)


def publish_sync(df):
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "Date": row["Date"].isoformat(),
                "Close": float(row["Close"]),
                "m_force": float(row["m_force"]),
            }
        )
    payload = {"batch": True, "rows": rows}
    publish.single(TOPIC, json.dumps(payload), hostname=MQTT_HOST, port=MQTT_PORT)


import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fake_df(n=30):
    """產生假資料 DataFrame"""
    now = datetime.now()
    dates = [now - timedelta(minutes=n - i) for i in range(n)]

    df = pd.DataFrame(
        {
            "Date": dates,
            "Close": np.random.normal(100, 2, n),
            "m_force": np.random.normal(0, 1, n),
        }
    ).set_index("Date")

    return df


if __name__ == "__main__":
    print("⏳ 生成假資料 & MQTT 測試中...")

    df = fake_df(30)

    # === Step 1: Batch 初次同步 ===
    print("📤 publish_sync() ...")
    publish_sync(df)

    # === Step 2: 實時更新 ===
    print("🚀 模擬即時更新 start")
    for i in range(20):
        new_date = datetime.now()
        new_close = 100 + np.sin(i / 3) * 2 + np.random.normal(0, 0.3)
        new_force = np.random.normal(0, 1)

        print(f"update {i}: Close={new_close:.2f}, m_force={new_force:.2f}")

        publish_update(new_date, new_close, new_force)
        time.sleep(1)

    print("🎉 測試完成！")
