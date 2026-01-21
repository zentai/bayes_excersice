from __future__ import annotations

import json
from typing import Protocol, Optional
import pandas as pd


# ============================================================
# Interface（TradingDashboard 只依賴這個）
# ============================================================


class DashboardPublisher(Protocol):
    def publish_snapshot(self, df: pd.DataFrame) -> None: ...
    def publish_tick(self, row: pd.Series) -> None: ...
    def start(self) -> None: ...


# ============================================================
# Payload helpers（仍然屬於 transport）
# ============================================================


def _row_to_payload(idx, row: pd.Series) -> dict:
    date = row["Date"] if "Date" in row else idx
    date_str = date.isoformat() if hasattr(date, "isoformat") else str(date)

    def _get(name, default):
        return row[name] if name in row and pd.notna(row[name]) else default

    payload = {
        "Date": date_str,
        "Close": float(_get("Close", 0.0)),
        "m_pc": float(_get("m_pc", 0.0)),
        "m_force": float(_get("m_force", 0.0)),
        "m_force_trend": float(_get("m_force_trend", 0.0)),
        "m_pt_speed": float(_get("m_pt_speed", 0.0)),
        "m_regime_noise_level": float(_get("m_regime_noise_level", 0)),
        "HMM_Signal": int(_get("HMM_Signal", 0)),
        "bocpd_cp_excess": float(_get("bocpd_cp_excess", 0.0)),
        "bocpd_risk": float(_get("bocpd_risk", 0.0)),
        "z_srl": float(_get("z_srl", 0.0)),
        "srl_eff": float(_get("srl_eff", 0.0)),
    }

    order_val: Optional[object] = "Buy" if _get("BuySignal", None) else None
    if _get("ShortSignal", None):
        order_val = "Short"
    if isinstance(order_val, dict):
        payload["order"] = order_val
    elif isinstance(order_val, str) and order_val:
        payload["order"] = {"side": order_val}
    else:
        payload["order"] = None

    return payload


# ============================================================
# MQTT Publisher（含 resync callback）
# ============================================================


class MqttDashboardPublisher:
    """
    - 負責 publish snapshot / tick
    - 監聽 resync request
    - callback 只會呼叫 dashboard.sync()
    """

    def __init__(
        self,
        symbol: str,
        interval: str,
        host: str = "localhost",
        port: int = 1883,
    ):
        self.symbol = symbol.lower()
        self.interval = interval
        self.host = host
        self.port = port

        self.data_topic = f"mkt/{self.symbol}/{self.interval}/k_channel"
        self.req_topic = f"mkt/{self.symbol}/{self.interval}/request"

        self._dashboard = None  # late bind

        # lazy import（避免没装 mqtt 直接炸）
        import paho.mqtt.client as mqtt
        import paho.mqtt.publish as publish

        self._mqtt = mqtt
        self._publish = publish

        self._client = mqtt.Client(
            client_id=f"dashboard-publisher-{self.symbol}-{self.interval}"
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    # ---------- binding ----------

    def bind_dashboard(self, dashboard):
        """
        Publisher 反向持有 dashboard reference（这是唯一允许的反向）
        """
        self._dashboard = dashboard

    # ---------- lifecycle ----------

    def start(self):
        self._client.connect(self.host, self.port)
        self._client.loop_start()

    # ---------- mqtt callbacks ----------

    def _on_connect(self, client, userdata, flags, rc):
        client.subscribe(self.req_topic)

    def _on_message(self, client, userdata, msg):
        """
        sync callback 落点
        """
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return

        if payload.get("type") != "sync":
            return

        if self._dashboard is None:
            return

        # ⭐ 唯一的反向动作：触发 sync
        self._dashboard.sync()

    # ---------- publish ----------

    def publish_snapshot(self, df: pd.DataFrame):
        rows = [_row_to_payload(idx, row) for idx, row in df.iterrows()]
        payload = {
            "batch": True,
            "rows": rows,
        }
        self._publish.single(
            self.data_topic,
            json.dumps(payload),
            hostname=self.host,
            port=self.port,
        )

    def publish_tick(self, row: pd.Series):
        payload = _row_to_payload(row.name, row)
        self._publish.single(
            self.data_topic,
            json.dumps(payload),
            hostname=self.host,
            port=self.port,
        )


# ============================================================
# No-Op Publisher（MQTT 不存在时）
# ============================================================


class NullDashboardPublisher:
    def bind_dashboard(self, dashboard): ...
    def start(self): ...
    def publish_snapshot(self, df: pd.DataFrame): ...
    def publish_tick(self, row: pd.Series): ...


class TradingDashboard:
    """
    Pure projection service.

    - 持有一份 df snapshot
    - 所有 publish 都基于 internal memory
    - 不知道 MQTT / topic / UI / resync
    """

    def __init__(self, publisher: DashboardPublisher):
        self._publisher = publisher
        self._df: Optional[pd.DataFrame] = None

        # 允许 publisher 反向绑定（仅此一次）
        if hasattr(self._publisher, "bind_dashboard"):
            self._publisher.bind_dashboard(self)

    # ---------- state ----------
    def update(self, df: pd.DataFrame):
        self._df = df.copy()

    # ---------- intent ----------

    def sync(self):
        if self._df is None or self._df.empty:
            return
        self._publisher.publish_snapshot(self._df[200:])

    def tick(self):
        if self._df is None or self._df.empty:
            return
        self._publisher.publish_tick(self._df.iloc[-1])


def build_mqtt_trading_dashboard(df, params):
    # 尝试启用 MQTT
    try:
        publisher = MqttDashboardPublisher(
            symbol=params.symbol.name,
            interval=params.interval,
        )
    except Exception as e:
        print(f"No MQTT install, error: {e}")
        publisher = NullDashboardPublisher()

    dashboard = TradingDashboard(publisher)
    dashboard.update(df)

    publisher.start()  # 监听 resync
    return dashboard
