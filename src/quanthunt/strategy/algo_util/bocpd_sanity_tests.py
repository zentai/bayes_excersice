"""
BOCPD Event-Based Sanity Tests
=============================

目标：
- 不测试 cp 数值
- 不测试 timing 精确度
- 只测试「事件有没有在合理阶段出现」

层级严格分离：
1) BOCPD：只产出 signal（你已有）
2) Event Detector：policy（本文件）
3) World Generator：ground truth
4) Assertions：语义级

适用于：
- Gaussian G0（regime death）
- Student-t P1（risk buffer）
"""

from __future__ import annotations
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Tuple
from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Callable
import csv


# -----------------------------
# Evidence Row
# -----------------------------
@dataclass
class EvidenceRow:
    t: int
    x: float

    # G0 core
    g0_cp: float
    g0_rl_mean: float
    g0_rl_mode: int
    g0_support: int

    # G0 evidence
    g0_log_pred_mix: float
    g0_surprise: float
    g0_surprise_ewma: float
    g0_z_mix: float
    g0_z2_ewma: float
    g0_scale_mix: float
    g0_conf: float

    # P1 risk (optional)
    p1_risk: float
    p1_scale_mix: float
    p1_shock: float
    p1_tail: float

    # events
    g0_event: Any
    p1_event: Any

    # helpful flags for quick eyeballing
    flag_surprise_hi: bool
    flag_z2_hi: bool
    flag_conf_low: bool


# -----------------------------
# Dump helpers
# -----------------------------
def dump_rows_to_csv(rows: List[EvidenceRow], path: str) -> None:
    if not rows:
        print("[dump] no rows to write")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"[dump] wrote {len(rows)} rows -> {path}")


def print_rows(rows: List[EvidenceRow], head: int = 10) -> None:
    # compact console view (no pandas dependency)
    for r in rows[:head]:
        print(
            f"t={r.t:4d} x={r.x:+.4f} "
            f"g0_ev={str(r.g0_event):>22} cp={r.g0_cp:.4f} rl_mode={r.g0_rl_mode:3d} sup={r.g0_support:3d} "
            f"spr={r.g0_surprise:.3f} sprE={r.g0_surprise_ewma:.3f} z={r.g0_z_mix:+.2f} z2E={r.g0_z2_ewma:.3f} "
            f"conf={r.g0_conf:.3f} | "
            f"p1_ev={str(r.p1_event):>14} risk={r.p1_risk:.4f} shock={r.p1_shock:.3f} tail={r.p1_tail:.3f}"
            f"{' [S!]' if r.flag_surprise_hi else ''}"
            f"{' [Z2!]' if r.flag_z2_hi else ''}"
            f"{' [C↓]' if r.flag_conf_low else ''}"
        )


def window_around_first_bad(
    rows: List[EvidenceRow], is_bad: Callable[[EvidenceRow], bool], w: int = 20
) -> List[EvidenceRow]:
    for i, r in enumerate(rows):
        if is_bad(r):
            lo = max(0, i - w)
            hi = min(len(rows), i + w + 1)
            return rows[lo:hi]
    return []


# -----------------------------
# The main capture function
# -----------------------------
def capture_evidence(
    xs: List[float],
    g0_engine: Callable[[float], Any],  # returns BOCPDOutputs
    p1_engine: Optional[Callable[[float], Any]],
    classify_events: Callable[
        [Any, Optional[Any]], Any
    ],  # returns EventFrame or tuple(g0_event,p1_event)
    # thresholds only for flags (not for logic)
    th_surprise_ewma: float = 2.0,
    th_z2_ewma: float = 4.0,
    th_conf_low: float = 0.35,
) -> List[EvidenceRow]:
    rows: List[EvidenceRow] = []
    for t, x in enumerate(xs):
        out_g0 = g0_engine(float(x))
        out_p1 = p1_engine(float(x)) if p1_engine is not None else None

        ev = classify_events(out_g0, out_p1)
        # support both styles:
        if hasattr(ev, "g0_event"):
            g0_ev = ev.g0_event
            p1_ev = ev.p1_event
        else:
            g0_ev, p1_ev = ev

        # P1 fields fallback
        if out_p1 is None:
            p1_risk = 0.0
            p1_scale = float("nan")
            p1_shock = float("nan")
            p1_tail = float("nan")
        else:
            p1_risk = float(getattr(out_p1, "risk_level", 0.0))
            p1_scale = float(getattr(out_p1, "scale_mix", float("nan")))
            p1_shock = float(getattr(out_p1, "shock_score", float("nan")))
            p1_tail = float(getattr(out_p1, "tail_prob_k", float("nan")))

        r = EvidenceRow(
            t=t,
            x=float(x),
            g0_cp=float(out_g0.cp_prob),
            g0_rl_mean=float(out_g0.run_length_mean),
            g0_rl_mode=int(out_g0.run_length_mode),
            g0_support=int(out_g0.support),
            g0_log_pred_mix=float(out_g0.log_pred_mix),
            g0_surprise=float(out_g0.surprise),
            g0_surprise_ewma=float(out_g0.surprise_ewma),
            g0_z_mix=float(out_g0.z_mix),
            g0_z2_ewma=float(out_g0.z2_ewma),
            g0_scale_mix=float(out_g0.scale_mix),
            g0_conf=float(out_g0.regime_confidence),
            p1_risk=float(p1_risk),
            p1_scale_mix=float(p1_scale),
            p1_shock=float(p1_shock),
            p1_tail=float(p1_tail),
            g0_event=g0_ev,
            p1_event=p1_ev,
            flag_surprise_hi=float(out_g0.surprise_ewma) >= th_surprise_ewma,
            flag_z2_hi=float(out_g0.z2_ewma) >= th_z2_ewma,
            flag_conf_low=float(out_g0.regime_confidence) <= th_conf_low,
        )
        rows.append(r)
    return rows


# =====================================================
# Event Definitions
# =====================================================


class G0Event(Enum):
    NONE = auto()
    REGIME_DRIFT_AWAKENING = auto()
    REGIME_DEATH_CANDIDATE = auto()
    REGIME_DEATH_CONFIRMED = auto()


class P1Event(Enum):
    NONE = auto()
    SHOCK_DETECTED = auto()
    RISK_ACCUMULATION = auto()
    RISK_NORMALIZED = auto()


@dataclass(frozen=True)
class EventFrame:
    t: int
    g0_event: G0Event
    p1_event: P1Event


# =====================================================
# Event Detectors (POLICY LAYER)
# =====================================================


class G0EventDetector:
    """
    Gaussian BOCPD policy:
    - 不依赖 hazard
    - 不用绝对 cp
    - 先 drift，再 death
    """

    def __init__(
        self,
        drift_surprise_th=0.8,
        cp_candidate_th=0.35,
        cp_confirm_th=0.55,
        confirm_len=5,
    ):
        self.drift_surprise_th = drift_surprise_th
        self.cp_candidate_th = cp_candidate_th
        self.cp_confirm_th = cp_confirm_th
        self.confirm_len = confirm_len
        self._confirm_streak = 0

    def detect(self, out) -> G0Event:
        # Phase 2 / 3: drift awakening
        if out.surprise_ewma > self.drift_surprise_th:
            return G0Event.REGIME_DRIFT_AWAKENING

        # Phase 4: death
        if out.cp_prob > self.cp_candidate_th:
            if out.cp_prob > self.cp_confirm_th:
                self._confirm_streak += 1
            else:
                self._confirm_streak = 0

            if self._confirm_streak >= self.confirm_len:
                return G0Event.REGIME_DEATH_CONFIRMED

            return G0Event.REGIME_DEATH_CANDIDATE

        self._confirm_streak = 0
        return G0Event.NONE


class P1EventDetector:
    """
    Student-t BOCPD policy:
    - 只负责风险
    - 不碰 regime
    """

    def __init__(
        self,
        shock_th=0.7,
        risk_th=0.4,
        normalize_th=0.15,
    ):
        self.shock_th = shock_th
        self.risk_th = risk_th
        self.normalize_th = normalize_th

    def detect(self, out) -> P1Event:
        if out.shock_score > self.shock_th:
            return P1Event.SHOCK_DETECTED
        if out.risk_level > self.risk_th:
            return P1Event.RISK_ACCUMULATION
        if out.risk_level < self.normalize_th:
            return P1Event.RISK_NORMALIZED
        return P1Event.NONE


# =====================================================
# Worlds (GROUND TRUTH GENERATORS)
# =====================================================


def world_stable(n=800, sigma=1.0, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=n)


def world_slow_drift(n=800, drift=0.01, sigma=1.0, seed=2):
    rng = np.random.default_rng(seed)
    xs, mu = [], 0.0
    for _ in range(n):
        mu += drift
        xs.append(mu + rng.normal(0, sigma))
    return xs


def world_regime_break(n1=400, n2=400, mu_jump=8.0, sigma=1.0, seed=3):
    rng = np.random.default_rng(seed)
    xs = list(rng.normal(0.0, sigma, size=n1))
    xs += list(rng.normal(mu_jump, sigma, size=n2))
    return xs


def world_dirty_trend(n=800, drift=0.01, shock_prob=0.03, sigma=1.0, seed=4):
    rng = np.random.default_rng(seed)
    xs, mu = [], 0.0
    for _ in range(n):
        mu += drift
        x = mu + rng.normal(0, sigma)
        if rng.random() < shock_prob:
            x += rng.normal(0, 6.0)
        xs.append(x)
    return xs


# =====================================================
# Runner
# =====================================================


def run_world(xs, duo) -> List[EventFrame]:
    g0_det = G0EventDetector()
    p1_det = P1EventDetector()

    events = []
    for t, x in enumerate(xs):
        out_g0, out_p1 = duo.update(float(x))
        events.append(
            EventFrame(
                t=t,
                g0_event=g0_det.detect(out_g0),
                p1_event=p1_det.detect(out_p1),
            )
        )
    return events


# =====================================================
# Assertions (SEMANTIC)
# =====================================================


def assert_no_g0_events(events):
    bad = [e for e in events if e.g0_event != G0Event.NONE]
    assert not bad, f"G0 should be silent but got {bad[:5]}"


def assert_drift_awakening(events):
    awakens = [e for e in events if e.g0_event == G0Event.REGIME_DRIFT_AWAKENING]
    assert awakens, "No drift awakening detected"


def assert_regime_death(events):
    deaths = [e for e in events if e.g0_event == G0Event.REGIME_DEATH_CONFIRMED]
    assert deaths, "No regime death confirmed"


def assert_p1_shock(events):
    shocks = [e for e in events if e.p1_event == P1Event.SHOCK_DETECTED]
    assert shocks, "No P1 shock detected"


# =====================================================
# Replay Plot (ASCII)
# =====================================================


def event_replay_plot(events: List[EventFrame], width=120):
    print("\nEvent Replay Timeline\n" + "-" * width)
    for e in events:
        if e.g0_event != G0Event.NONE or e.p1_event != P1Event.NONE:
            print(
                f"t={e.t:04d} | " f"G0={e.g0_event.name:<24} | " f"P1={e.p1_event.name}"
            )
    print("-" * width)


def classify_events(out_g0, out_p1):
    # G0：只要不是 NONE，我们就当作“有问题”
    if out_g0.surprise_ewma > 1.0:
        g0_event = "G0_DRIFT"
    elif out_g0.cp_prob > 0.4:
        g0_event = "G0_DEATH_CANDIDATE"
    else:
        g0_event = "G0_NONE"

    # P1：只做记录，不参与判断
    if out_p1 is None:
        p1_event = "P1_NONE"
    elif out_p1.shock_score > 0.7:
        p1_event = "P1_SHOCK"
    elif out_p1.risk_level > 0.4:
        p1_event = "P1_RISK"
    else:
        p1_event = "P1_NONE"

    return g0_event, p1_event


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    # 你已有的 BOCPD
    from bocpdz import BOCPDGaussianG0, BOCPDStudentTP1, DualBOCPD

    g0 = BOCPDGaussianG0(hazard=0.00083)
    p1 = BOCPDStudentTP1(hazard=0.00083)
    duo = DualBOCPD(g0, p1)

    print("\n=== Stable World ===")
    xs = world_stable()
    ev = run_world(xs, duo)

    rows = capture_evidence(
        xs=xs,
        g0_engine=g0.update,
        p1_engine=p1.update,  # 沒有就傳 None
        classify_events=classify_events,  # 你現有的 event 判定器
        th_surprise_ewma=1.0,  # 用你 policy 內的
        th_z2_ewma=4.0,
        th_conf_low=0.35,
    )

    # 1) 直接看前 30 行（通常 drift hallucination 很早就開始）
    print_rows(rows, head=30)

    # 2) 印出「第一個 G0 不該出現的 event」附近窗口
    win = window_around_first_bad(
        rows,
        is_bad=lambda r: str(r.g0_event) != "G0Event.NONE",  # 依你 enum 命名調整
        w=25,
    )
    print("\n--- window around first bad G0 event ---")
    print_rows(win, head=len(win))

    # 3) 落地成 CSV（你可以用 excel / pandas 開）
    dump_rows_to_csv(rows, "stable_world_evidence.csv")

    assert_no_g0_events(ev)
    event_replay_plot(ev)

    print("\n=== Slow Drift World ===")
    xs = world_slow_drift()
    ev = run_world(xs, duo)
    assert_drift_awakening(ev)
    event_replay_plot(ev)

    print("\n=== Regime Break World ===")
    xs = world_regime_break()
    ev = run_world(xs, duo)
    assert_regime_death(ev)
    event_replay_plot(ev)

    print("\n=== Dirty Trend World ===")
    xs = world_dirty_trend()
    ev = run_world(xs, duo)
    assert_p1_shock(ev)
    event_replay_plot(ev)

    print("\n✅ Event-based BOCPD sanity tests completed.")
