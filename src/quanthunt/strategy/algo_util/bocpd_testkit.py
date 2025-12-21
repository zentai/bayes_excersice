#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BOCPD Test Kit — Level 1 (Indicator Sanity)
------------------------------------------
生成 4 種世界 (Stable / Slow Drift / Shock / Regime Break)，
在已知 ground-truth world 下驗證 G0/P1 的「連續指標」行為是否 sane、穩定、可區分。

你需要做的事：
1) 把你的 G0/P1 實作接到 Adapter：update(x) -> dict 指標
2) 跑 runner 產生 CSV + plots + Level 1 asserts

不做的事：
- 不驗證 event mapping（Level 2）
- 不驗證策略/交易績效（Level 3）
- 不整合 policy/MOSAIC/HMM

依賴:
pip install numpy pandas matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bocpdz import BOCPDStudentTP1, BOCPDGaussianG0

# ----------------------------
# 0) 指標 row schema（統一）
# ----------------------------

G0_FIELDS = [
    "g0_cp_prob",
    "g0_run_length_mode",
    "g0_surprise_ewma",
    "g0_z2_ewma",
    "g0_confidence",
]

P1_FIELDS = [
    "p1_risk_level",
    "p1_shock_score",
    "p1_tail_prob",
    "p1_z",
]

OPTIONAL_EVENT_FIELDS = ["g0_event", "p1_event"]

ROW_FIELDS = (
    ["t", "world_id", "seed", "x"] + G0_FIELDS + P1_FIELDS + OPTIONAL_EVENT_FIELDS
)


# ----------------------------
# 1) World Generators
# ----------------------------


@dataclass
class WorldConfig:
    name: str
    T: int = 2000
    seed: int = 42

    # base noise
    mu: float = 0.0
    sigma: float = 1.0

    # drift
    drift_per_tick: float = 0.0005  # Slow Drift

    # shock (mixture)
    shock_prob: float = 0.01
    shock_scale: float = 8.0  # shock magnitude in sigma units
    shock_sign: str = "both"  # 'both'/'pos'/'neg'

    # regime break
    break_t: int = 800
    mu2: float = 3.0
    sigma2: Optional[float] = None  # if set, variance shift


class WorldGenerator:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def generate(self) -> np.ndarray:
        raise NotImplementedError


class StableWorld(WorldGenerator):
    """x_t = mu + eps_t, eps light-tail (Gaussian), bounded variance, no structural change."""

    def generate(self) -> np.ndarray:
        cfg = self.cfg
        eps = self.rng.normal(0.0, cfg.sigma, size=cfg.T)
        x = np.clip(cfg.mu + eps, cfg.mu - 3 * cfg.sigma, cfg.mu + 3 * cfg.sigma)
        return x


class SlowDriftWorld(WorldGenerator):
    """x_t = mu_t + eps_t, mu_t drifts slowly, eps light-tail."""

    def generate(self) -> np.ndarray:
        cfg = self.cfg
        t = np.arange(cfg.T, dtype=float)
        mu_t = cfg.mu + cfg.drift_per_tick * t
        eps = self.rng.normal(0.0, cfg.sigma, size=cfg.T)
        x = mu_t + eps
        return x


class ShockWorld(WorldGenerator):
    """
    x_t = mu + eps_t + s_t
    eps light-tail; s_t sparse shocks (fat-tail-like via mixture).
    """

    def generate(self) -> np.ndarray:
        cfg = self.cfg
        eps = self.rng.normal(0.0, cfg.sigma, size=cfg.T)
        shocks = np.zeros(cfg.T, dtype=float)

        mask = self.rng.uniform(0, 1, size=cfg.T) < cfg.shock_prob
        # shock amplitude ~ Normal(0, shock_scale*sigma) but sparse => heavy tails in mixture sense
        amp = self.rng.normal(0.0, cfg.shock_scale * cfg.sigma, size=cfg.T)

        if cfg.shock_sign == "pos":
            amp = np.abs(amp)
        elif cfg.shock_sign == "neg":
            amp = -np.abs(amp)

        shocks[mask] = amp[mask]
        x = cfg.mu + eps + shocks
        return x


class RegimeBreakWorld(WorldGenerator):
    """
    Persistent distribution switch at t=break_t.
    mean shift and/or variance shift.
    """

    def generate(self) -> np.ndarray:
        cfg = self.cfg
        T = cfg.T
        bt = int(cfg.break_t)
        bt = max(1, min(bt, T - 1))

        sigma1 = cfg.sigma
        sigma2 = cfg.sigma2 if cfg.sigma2 is not None else cfg.sigma

        eps1 = self.rng.normal(0.0, sigma1, size=bt)
        eps2 = self.rng.normal(0.0, sigma2, size=T - bt)

        x1 = cfg.mu + eps1
        x2 = cfg.mu2 + eps2
        x = np.concatenate([x1, x2], axis=0)
        return x


def make_world(cfg: WorldConfig) -> WorldGenerator:
    name = cfg.name.lower()
    if name in ["stable", "stable_world"]:
        return StableWorld(cfg)
    if name in ["drift", "slow_drift", "slowdrift", "slow_drift_world"]:
        return SlowDriftWorld(cfg)
    if name in ["shock", "shock_world"]:
        return ShockWorld(cfg)
    if name in ["break", "regime", "regime_break", "regimebreak", "regime_break_world"]:
        return RegimeBreakWorld(cfg)
    raise ValueError(f"Unknown world name: {cfg.name}")


# ----------------------------
# 2) Model Adapter（接你的 G0 / P1）
# ----------------------------


class BOCPDAdapter:
    """
    你需要把自己的 G0/P1 實作接進來：
    - self.g0.update(x) -> dict
    - self.p1.update(x) -> dict

    dict 必須至少包含：
    G0: cp_prob, run_length_mode, surprise_ewma, z2_ewma, confidence
    P1: risk_level, shock_score, tail_prob, z
    事件欄位可選：event
    """

    def __init__(self, g0: Any, p1: Any):
        self.g0 = g0
        self.p1 = p1

    def reset(self):
        # 視你的實作而定；沒有就略過
        for obj in [self.g0, self.p1]:
            if hasattr(obj, "reset") and callable(obj.reset):
                obj.reset()

    def update(self, x: float) -> Dict[str, Any]:
        g0_out = self.g0.update(x)
        p1_out = self.p1.update(x)

        row = {
            "g0_cp_prob": float(g0_out.cp_prob),
            "g0_run_length_mode": float(g0_out.run_length_mode),
            "g0_surprise_ewma": float(g0_out.surprise_ewma),
            "g0_z2_ewma": float(g0_out.z2_ewma),
            "g0_confidence": float(g0_out.regime_confidence),
            "p1_shock_score": float(p1_out.shock_score),
            "p1_risk_level": float(p1_out.risk_level),
            "p1_tail_prob": float(p1_out.tail_prob_k),
            "p1_z": float(p1_out.z_mix),
        }

        return row


# ----------------------------
# 2.1) （可選）Dummy 模型：讓你先跑起來
#      你之後換成真實 G0/P1
# ----------------------------


class DummyG0:
    """
    這不是 BOCPD，只是為了讓 testkit 可跑、看 pipeline 是否正常。
    請替換成你的 Gaussian BOCPD。
    """

    def __init__(self, ewma_alpha=0.02):
        self.alpha = ewma_alpha
        self.t = 0
        self.mean = 0.0
        self.var = 1.0
        self.surprise = 0.0
        self.z2 = 0.0
        self.runlen = 0.0

    def reset(self):
        self.__init__(self.alpha)

    def update(self, x: float) -> Dict[str, Any]:
        self.t += 1
        # naive EWMA mean/var
        dx = x - self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var = (1 - self.alpha) * self.var + self.alpha * (dx * dx + 1e-9)

        z = dx / (math.sqrt(self.var) + 1e-9)
        z2 = z * z

        self.surprise = (1 - self.alpha) * self.surprise + self.alpha * abs(z)
        self.z2 = (1 - self.alpha) * self.z2 + self.alpha * z2

        # fake cp_prob: increases if surprise is high
        cp_prob = 1 / (1 + math.exp(-(self.surprise - 2.0)))
        # fake run length mode: grows unless cp_prob too high
        if cp_prob > 0.7:
            self.runlen = max(0.0, self.runlen * 0.3)
        else:
            self.runlen = self.runlen + 1.0

        confidence = max(0.0, 1.0 - min(1.0, self.surprise / 5.0))

        return dict(
            cp_prob=cp_prob,
            run_length_mode=self.runlen,
            surprise_ewma=self.surprise,
            z2_ewma=self.z2,
            confidence=confidence,
            event="DUMMY",
        )


class DummyP1:
    """
    這不是 BOCPD，只是為了 pipeline 可跑。
    請替換成你的 Student-t BOCPD。
    """

    def __init__(self, ewma_alpha=0.05):
        self.alpha = ewma_alpha
        self.mean = 0.0
        self.var = 1.0
        self.risk = 0.0
        self.tail = 0.0
        self.shock = 0.0

    def reset(self):
        self.__init__(self.alpha)

    def update(self, x: float) -> Dict[str, Any]:
        dx = x - self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var = (1 - self.alpha) * self.var + self.alpha * (dx * dx + 1e-9)
        z = dx / (math.sqrt(self.var) + 1e-9)

        # fake tail prob: sigmoid(|z|-3)
        tail_prob = 1 / (1 + math.exp(-(abs(z) - 3.0)))
        shock_score = tail_prob * abs(z)

        self.tail = (1 - self.alpha) * self.tail + self.alpha * tail_prob
        self.shock = (1 - self.alpha) * self.shock + self.alpha * shock_score
        self.risk = (1 - self.alpha) * self.risk + self.alpha * min(
            1.0, self.shock / 5.0
        )

        return dict(
            risk_level=self.risk,
            shock_score=self.shock,
            tail_prob=self.tail,
            z=z,
            event="DUMMY",
        )


# ----------------------------
# 3) Runner：world -> model -> dataframe -> csv
# ----------------------------


@dataclass
class RunConfig:
    out_dir: str = "bocpd_testkit_out"
    make_plots: bool = True
    overwrite: bool = True


def run_world(
    adapter: BOCPDAdapter, world_id: str, x: np.ndarray, seed: int
) -> pd.DataFrame:
    adapter.reset()
    rows: List[Dict[str, Any]] = []
    for t, xt in enumerate(x):
        out = adapter.update(float(xt))
        row = {
            "t": int(t),
            "world_id": str(world_id),
            "seed": int(seed),
            "x": float(xt),
        }
        row.update(out)
        # ensure optional fields exist
        if "g0_event" not in row:
            row["g0_event"] = ""
        if "p1_event" not in row:
            row["p1_event"] = ""
        rows.append(row)

    df = pd.DataFrame(rows, columns=ROW_FIELDS)
    return df


# ----------------------------
# 4) Level 1 Asserts（統計型）
# ----------------------------


@dataclass
class AssertConfig:
    # generic sanity bounds (adjust to your model's true ranges)
    # NOTE: if you know exact bounds, tighten them.
    cp_prob_min: float = 0.0
    cp_prob_max: float = 1.0
    conf_min: float = 0.0
    conf_max: float = 1.0
    tail_prob_min: float = 0.0
    tail_prob_max: float = 1.0
    risk_min: float = 0.0
    risk_max: float = 1.0

    # Stable expectations (tune these thresholds to your system)
    stable_cp_prob_q95_max: float = 0.2
    stable_tail_q95_max: float = 0.2
    stable_risk_q95_max: float = 0.25
    stable_runlen_median_min: float = 200.0  # depends on T + hazard; tune
    stable_conf_median_min: float = 0.4  # tune to your confidence scale

    # Shock expectations
    shock_tail_q99_min: float = 0.3  # should spike sometimes
    shock_cp_prob_q95_max: float = 0.6  # G0 should not stay too high (tune)
    shock_runlen_median_min: float = 50.0  # should not collapse frequently

    # Break expectations
    break_cp_prob_post_q90_min: float = 0.4  # post-break should show higher cp_prob
    break_runlen_post_median_max: float = (
        200.0  # post-break median runlen lower early (tune)
    )
    break_surprise_post_median_min: float = 1.0  # mismatch rises (tune)

    # Drift expectations
    drift_surprise_slope_min: float = 0.0  # positive-ish (trend)
    drift_cp_prob_q95_max: float = 0.7  # should not look like hard break (tune)

    # warmup ticks to ignore in stats
    warmup: int = 50


class Level1AssertError(AssertionError):
    pass


def _check_bounds(df: pd.DataFrame, ac: AssertConfig) -> List[str]:
    errs = []

    def bad(msg):
        errs.append(msg)

    # cp_prob bounds
    if not (
        (df["g0_cp_prob"] >= ac.cp_prob_min) & (df["g0_cp_prob"] <= ac.cp_prob_max)
    ).all():
        bad("g0_cp_prob out of bounds")
    if not (
        (df["g0_confidence"] >= ac.conf_min) & (df["g0_confidence"] <= ac.conf_max)
    ).all():
        bad("g0_confidence out of bounds")
    if not (
        (df["p1_tail_prob"] >= ac.tail_prob_min)
        & (df["p1_tail_prob"] <= ac.tail_prob_max)
    ).all():
        bad("p1_tail_prob out of bounds")
    if not (
        (df["p1_risk_level"] >= ac.risk_min) & (df["p1_risk_level"] <= ac.risk_max)
    ).all():
        bad("p1_risk_level out of bounds")

    # finite
    for c in ["x"] + G0_FIELDS + P1_FIELDS:
        if not np.isfinite(df[c].to_numpy()).all():
            bad(f"{c} has NaN/Inf")

    return errs


def assert_stable(df: pd.DataFrame, ac: AssertConfig) -> List[str]:
    errs = []
    d = df.iloc[ac.warmup :].copy()

    q95_cp = d["g0_cp_prob"].quantile(0.95)
    if q95_cp > ac.stable_cp_prob_q95_max:
        errs.append(
            f"[Stable] g0_cp_prob q95 too high: {q95_cp:.4f} > {ac.stable_cp_prob_q95_max}"
        )

    q95_tail = d["p1_tail_prob"].quantile(0.95)
    if q95_tail > ac.stable_tail_q95_max:
        errs.append(
            f"[Stable] p1_tail_prob q95 too high: {q95_tail:.4f} > {ac.stable_tail_q95_max}"
        )

    q95_risk = d["p1_risk_level"].quantile(0.95)
    if q95_risk > ac.stable_risk_q95_max:
        errs.append(
            f"[Stable] p1_risk_level q95 too high: {q95_risk:.4f} > {ac.stable_risk_q95_max}"
        )

    med_run = d["g0_run_length_mode"].median()
    if med_run < ac.stable_runlen_median_min:
        errs.append(
            f"[Stable] g0_run_length_mode median too low: {med_run:.2f} < {ac.stable_runlen_median_min}"
        )

    med_conf = d["g0_confidence"].median()
    if med_conf < ac.stable_conf_median_min:
        errs.append(
            f"[Stable] g0_confidence median too low: {med_conf:.4f} < {ac.stable_conf_median_min}"
        )

    # "絕對不該高頻 collapse"（用 runlen 低值比例近似）
    low_run_ratio = (d["g0_run_length_mode"] < 10).mean()
    if low_run_ratio > 0.05:
        errs.append(
            f"[Stable] run_length low(<10) ratio too high: {low_run_ratio:.3%} > 5%"
        )

    return errs


def assert_drift(df: pd.DataFrame, ac: AssertConfig) -> List[str]:
    errs = []
    d = df.iloc[ac.warmup :].copy()

    # surprise should trend upward (粗略用線性回歸斜率)
    t = d["t"].to_numpy().astype(float)
    y = d["g0_surprise_ewma"].to_numpy().astype(float)
    t = t - t.mean()
    slope = float((t @ (y - y.mean())) / (t @ t + 1e-12))

    if slope < ac.drift_surprise_slope_min:
        errs.append(
            f"[Drift] g0_surprise_ewma slope too low: {slope:.6f} < {ac.drift_surprise_slope_min}"
        )

    q95_cp = d["g0_cp_prob"].quantile(0.95)
    if q95_cp > ac.drift_cp_prob_q95_max:
        errs.append(
            f"[Drift] g0_cp_prob q95 too high (looks like hard break): {q95_cp:.4f} > {ac.drift_cp_prob_q95_max}"
        )

    # P1 should not be heavily triggered
    q95_tail = d["p1_tail_prob"].quantile(0.95)
    if q95_tail > 0.6:
        errs.append(
            f"[Drift] p1_tail_prob q95 too high (treating drift as outlier): {q95_tail:.4f} > 0.6"
        )

    return errs


def assert_shock(df: pd.DataFrame, ac: AssertConfig) -> List[str]:
    errs = []
    d = df.iloc[ac.warmup :].copy()

    # P1 should have spikes sometimes
    q99_tail = d["p1_tail_prob"].quantile(0.99)
    if q99_tail < ac.shock_tail_q99_min:
        errs.append(
            f"[Shock] p1_tail_prob q99 too low (no shock response): {q99_tail:.4f} < {ac.shock_tail_q99_min}"
        )

    # G0 should not stay too high most of time
    q95_cp = d["g0_cp_prob"].quantile(0.95)
    if q95_cp > ac.shock_cp_prob_q95_max:
        errs.append(
            f"[Shock] g0_cp_prob q95 too high (overreacting): {q95_cp:.4f} > {ac.shock_cp_prob_q95_max}"
        )

    med_run = d["g0_run_length_mode"].median()
    if med_run < ac.shock_runlen_median_min:
        errs.append(
            f"[Shock] g0_run_length_mode median too low (frequent collapse): {med_run:.2f} < {ac.shock_runlen_median_min}"
        )

    return errs


def assert_break(df: pd.DataFrame, ac: AssertConfig, break_t: int) -> List[str]:
    errs = []
    d = df.iloc[ac.warmup :].copy()

    pre = d[d["t"] < break_t]
    post = d[d["t"] >= break_t]

    if len(pre) < 50 or len(post) < 50:
        errs.append(f"[Break] not enough samples around break_t={break_t}")
        return errs

    # post cp_prob should be higher
    post_q90_cp = post["g0_cp_prob"].quantile(0.90)
    if post_q90_cp < ac.break_cp_prob_post_q90_min:
        errs.append(
            f"[Break] post g0_cp_prob q90 too low: {post_q90_cp:.4f} < {ac.break_cp_prob_post_q90_min}"
        )

    # post run length mode should show collapse early (median low-ish)
    post_med_run = post["g0_run_length_mode"].median()
    if post_med_run > ac.break_runlen_post_median_max:
        errs.append(
            f"[Break] post g0_run_length_mode median too high (no collapse): {post_med_run:.2f} > {ac.break_runlen_post_median_max}"
        )

    post_med_sur = post["g0_surprise_ewma"].median()
    if post_med_sur < ac.break_surprise_post_median_min:
        errs.append(
            f"[Break] post g0_surprise_ewma median too low: {post_med_sur:.4f} < {ac.break_surprise_post_median_min}"
        )

    # Stable-like behavior should not dominate post (optional)
    post_low_cp_ratio = (post["g0_cp_prob"] < 0.1).mean()
    if post_low_cp_ratio > 0.9:
        errs.append(
            f"[Break] post cp_prob mostly low (model ignores break): {post_low_cp_ratio:.2%} low-cp ratio"
        )

    return errs


def run_level1_asserts(
    df: pd.DataFrame,
    world_name: str,
    ac: AssertConfig,
    break_t: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    errs = []
    errs.extend(_check_bounds(df, ac))

    w = world_name.lower()
    if "stable" in w:
        errs.extend(assert_stable(df, ac))
    elif "drift" in w:
        errs.extend(assert_drift(df, ac))
    elif "shock" in w:
        errs.extend(assert_shock(df, ac))
    elif "break" in w or "regime" in w:
        if break_t is None:
            errs.append("[Break] break_t is required for break world asserts")
        else:
            errs.extend(assert_break(df, ac, break_t=break_t))
    else:
        errs.append(f"Unknown world for asserts: {world_name}")

    return (len(errs) == 0, errs)


# ----------------------------
# 5) Plotting（最小高資訊量）
# ----------------------------


def _save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_world(df: pd.DataFrame, out_dir: str, tag: str, break_t: Optional[int] = None):
    os.makedirs(out_dir, exist_ok=True)

    t = df["t"].to_numpy()
    x = df["x"].to_numpy()

    # 1) x_t
    plt.figure(figsize=(12, 3))
    plt.plot(t, x)
    if break_t is not None:
        plt.axvline(break_t, linestyle="--")
    plt.title(f"{tag} — x_t")
    _save_fig(os.path.join(out_dir, f"{tag}_01_x.png"))

    # 2) G0 cp_prob + run_length
    plt.figure(figsize=(12, 4))
    plt.plot(t, df["g0_cp_prob"].to_numpy(), label="g0_cp_prob")
    plt.plot(t, df["g0_run_length_mode"].to_numpy(), label="g0_run_length_mode")
    if break_t is not None:
        plt.axvline(break_t, linestyle="--")
    plt.title(f"{tag} — G0 cp_prob & run_length_mode")
    plt.legend()
    _save_fig(os.path.join(out_dir, f"{tag}_02_g0_cp_runlen.png"))

    # 3) G0 mismatch panel
    plt.figure(figsize=(12, 4))
    plt.plot(t, df["g0_surprise_ewma"].to_numpy(), label="g0_surprise_ewma")
    plt.plot(t, df["g0_z2_ewma"].to_numpy(), label="g0_z2_ewma")
    plt.plot(t, df["g0_confidence"].to_numpy(), label="g0_confidence")
    if break_t is not None:
        plt.axvline(break_t, linestyle="--")
    plt.title(f"{tag} — G0 mismatch (surprise/z2/conf)")
    plt.legend()
    _save_fig(os.path.join(out_dir, f"{tag}_03_g0_mismatch.png"))

    # 4) P1 risk panel
    plt.figure(figsize=(12, 4))
    plt.plot(t, df["p1_risk_level"].to_numpy(), label="p1_risk_level")
    plt.plot(t, df["p1_tail_prob"].to_numpy(), label="p1_tail_prob")
    plt.plot(t, df["p1_shock_score"].to_numpy(), label="p1_shock_score")
    if break_t is not None:
        plt.axvline(break_t, linestyle="--")
    plt.title(f"{tag} — P1 risk (risk/tail/shock)")
    plt.legend()
    _save_fig(os.path.join(out_dir, f"{tag}_04_p1_risk.png"))


# ----------------------------
# 6) Test Kit 主程序（生成、跑、dump、assert、plot）
# ----------------------------


def run_testkit(
    adapter: BOCPDAdapter,
    worlds: List[WorldConfig],
    rc: RunConfig = RunConfig(),
    ac: AssertConfig = AssertConfig(),
) -> Dict[str, Any]:
    out_dir = rc.out_dir
    if rc.overwrite and os.path.exists(out_dir):
        # safety: only remove files we create (csv/png/json)
        for root, _, files in os.walk(out_dir):
            for f in files:
                if f.endswith((".csv", ".png", ".json")):
                    try:
                        os.remove(os.path.join(root, f))
                    except OSError:
                        pass
    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, Any] = {"passed": True, "worlds": []}

    for wc in worlds:
        gen = make_world(wc)
        x = gen.generate()

        df = run_world(adapter, world_id=wc.name, x=x, seed=wc.seed)

        tag = f"{wc.name}_seed{wc.seed}_T{wc.T}"
        csv_path = os.path.join(out_dir, f"{tag}.csv")
        df.to_csv(csv_path, index=False)

        break_t = (
            wc.break_t
            if wc.name.lower() in ["break", "regime_break", "regime"]
            else None
        )
        ok, errs = run_level1_asserts(df, wc.name, ac, break_t=break_t)

        if rc.make_plots:
            plot_world(df, out_dir=out_dir, tag=tag, break_t=break_t)

        world_report = {
            "world": wc.name,
            "seed": wc.seed,
            "T": wc.T,
            "csv": csv_path,
            "passed": bool(ok),
            "errors": errs,
            "break_t": break_t,
        }
        summary["worlds"].append(world_report)
        if not ok:
            summary["passed"] = False

    # write summary json
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ----------------------------
# 7) CLI / Demo entry
# ----------------------------


def default_worlds() -> List[WorldConfig]:
    return [
        WorldConfig(name="Stable", T=2000, seed=1, mu=0.0, sigma=0.15),
        WorldConfig(
            name="Drift", T=2000, seed=2, mu=0.0, sigma=1.0, drift_per_tick=0.1
        ),
        WorldConfig(
            name="Shock",
            T=2000,
            seed=3,
            mu=0.0,
            sigma=1.0,
            shock_prob=0.01,
            shock_scale=8.0,
        ),
        WorldConfig(
            name="Break",
            T=2000,
            seed=4,
            mu=0.0,
            sigma=3.0,
            break_t=900,
            mu2=12.0,
            drift_per_tick=0.9,
        ),
    ]


def main():
    # TODO: 換成你的真實模型
    g0 = BOCPDGaussianG0()
    p1 = BOCPDStudentTP1()
    adapter = BOCPDAdapter(g0=g0, p1=p1)

    worlds = default_worlds()
    rc = RunConfig(out_dir="bocpd_testkit_out", make_plots=True, overwrite=True)

    # TODO: 依你的指標尺度調整 AssertConfig 門檻
    ac = AssertConfig(
        warmup=50,
        stable_runlen_median_min=200.0,  # 你若 hazard 很高，這裡要降
        stable_cp_prob_q95_max=0.35,  # 你若 cp_prob 天生偏高，這裡要調
    )

    summary = run_testkit(adapter, worlds, rc, ac)

    print("\n=== BOCPD Test Kit Summary ===")
    print("PASS" if summary["passed"] else "FAIL")
    for w in summary["worlds"]:
        status = "PASS" if w["passed"] else "FAIL"
        print(f"- {w['world']:>6} [{status}]  csv={w['csv']}")
        if w["errors"]:
            for e in w["errors"]:
                print("   ", e)
    print(f"\nArtifacts in: {rc.out_dir}/ (csv, png, summary.json)\n")


if __name__ == "__main__":
    main()


# ----------------------------
# 8) （可選）pytest 入口
# ----------------------------
def test_level1_smoke():
    """
    pytest -q bocpd_testkit.py::test_level1_smoke
    這是 smoke test：確保 pipeline 能跑、輸出完整、且 dummy 模型不會崩。
    換成真實模型後，你就改成你的 adapter/g0/p1 初始化。
    """
    g0 = BOCPDGaussianG0()
    p1 = BOCPDStudentTP1()
    print("Updated g0 p1")
    adapter = BOCPDAdapter(g0=g0, p1=p1)
    worlds = default_worlds()
    rc = RunConfig(out_dir="bocpd_testkit_pytest_out", make_plots=False, overwrite=True)
    ac = AssertConfig(
        warmup=20, stable_runlen_median_min=50.0, stable_cp_prob_q95_max=0.6
    )
    summary = run_testkit(adapter, worlds, rc, ac)
    assert "worlds" in summary and len(summary["worlds"]) == 4
    # dummy 模型的 assert 門檻被放寬了，理論上應該 pass
    assert summary["passed"] is True
