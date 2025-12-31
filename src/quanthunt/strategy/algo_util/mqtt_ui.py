# mqtt_ui.py
import json
from typing import Optional

import pandas as pd
import paho.mqtt.publish as publish


MQTT_HOST = "localhost"
MQTT_PORT = 1883


def _build_topic(symbol: str, interval: str) -> str:
    """
    mkt/<symbol>/<interval>/k_channel
    symbol: BTC / XRP / ADA ...
    interval: 1m / 5m / 15m / 30m / 1hr / 4hr / 1day
    """
    return f"mkt/{symbol.lower()}/{interval}/k_channel"


def _row_to_payload(idx, row: pd.Series) -> dict:
    """
    將單一 row 轉成 UI 需要的 JSON 結構。
    必要欄位：
        Close, m_force
    選擇欄位（若無則給預設值）：
        regime, hmm_signal, cp_prob, risk, order
    """
    if hasattr(idx, "isoformat"):
        date_str = idx.isoformat()
    else:
        date_str = str(idx)

    def _get(name: str, default):
        return row[name] if name in row and pd.notna(row[name]) else default

    payload = {
        "Date": date_str,
        "Close": float(_get("Close", 0.0)),
        "m_force": float(_get("m_force", 0.0)),
        "regime": int(_get("regime", 0)),
        "hmm_signal": int(_get("hmm_signal", 0)),
        "cp_prob": float(_get("cp_prob", 0.0)),
        "risk": float(_get("risk", 0.0)),
    }

    order_val: Optional[object] = _get("order", None)
    if isinstance(order_val, dict):
        payload["order"] = order_val
    elif isinstance(order_val, str) and order_val:
        payload["order"] = {"side": order_val}
    else:
        payload["order"] = None

    return payload


def publish_update(symbol: str, interval: str, row: pd.Series):
    """
    單筆即時更新。

    參數：
        symbol  : 'BTC' / 'XRP' / 'ADA' ...
        interval: '1m' / '5m' / '15m' / '30m' / '1hr' / '4hr' / '1day'
        row     : DataFrame 的單一 row（含 Date index）
    """
    topic = _build_topic(symbol, interval)
    print(topic)
    payload = _row_to_payload(row.name, row)

    publish.single(
        topic,
        json.dumps(payload),
        hostname=MQTT_HOST,
        port=MQTT_PORT,
    )


def publish_sync(symbol: str, interval: str, df: pd.DataFrame):
    """
    批次同步：一次把 df 的所有 row 丟給 UI 做初始化。

    df:
        index = Date (DatetimeIndex)
        columns 至少包含 Close, m_force
        其他欄位同 _row_to_payload
    """
    topic = _build_topic(symbol, interval)

    rows = []
    for idx, row in df.iterrows():
        rows.append(_row_to_payload(idx, row))

    payload = {
        "batch": True,
        "rows": rows,
    }

    publish.single(
        topic,
        json.dumps(payload),
        hostname=MQTT_HOST,
        port=MQTT_PORT,
    )
