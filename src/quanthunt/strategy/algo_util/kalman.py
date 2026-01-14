# ============================================================
# Core imports
# ============================================================

import numpy as np
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Tuple, Optional
from quanthunt.config.core_config import config

ZERO = config.zero

# ============================================================
# Protocol definitions (interfaces)
# ============================================================


class StateModel(Protocol):
    def predict_state(self, x: float, ctx: Dict[str, Any]) -> float: ...


class ObservationModel(Protocol):
    def residual(self, x_pred: float, y: float, ctx: Dict[str, Any]) -> float: ...


class QRPolicy(Protocol):
    def get_QR(self, ctx: Dict[str, Any]) -> Tuple[float, float]: ...


class ResetPolicy(Protocol):
    def should_freeze(self, ctx: Dict[str, Any]) -> bool: ...

    def should_reset(self, ctx: Dict[str, Any]) -> bool: ...


# ============================================================
# Base scalar state-space engine (Kalman-style)
# ============================================================


@dataclass
class BaseStateSpaceEngine:
    x: Optional[float]
    P: float
    Q_base: float
    R_base: float

    state_model: StateModel
    obs_model: ObservationModel
    qr_policy: QRPolicy
    reset_policy: Optional[ResetPolicy] = None

    def step(self, y: float, ctx: Dict[str, Any]) -> Dict[str, Any]:
        # --- init ---
        if self.x is None:
            self.x = float(y)
            self.P = 1.0
            return self._pack(0.0, 0.0, y, 0.0, frozen=True)

        # --- reset / freeze gate ---
        if self.reset_policy:
            if self.reset_policy.should_reset(ctx):
                self.x = float(y)
                self.P = 1.0
                return self._pack(0.0, 0.0, y, 0.0, reset=True)

            if self.reset_policy.should_freeze(ctx):
                return self._pack(self.x, self.P, y, 0.0, frozen=True)

        # --- predict ---
        x_pred = self.state_model.predict_state(self.x, ctx)
        Q_t, R_t = self.qr_policy.get_QR(ctx)
        P_pred = self.P + Q_t

        # --- update ---
        resid = self.obs_model.residual(x_pred, y, ctx)
        S = P_pred + R_t
        K = P_pred / S

        self.x = x_pred + K * resid
        self.P = (1.0 - K) * P_pred

        return self._pack(x_pred, P_pred, y, resid, K=K, Q=Q_t, R=R_t)

    def _pack(self, x_pred, P_pred, y, resid, **flags):
        return {
            "x": self.x,
            "P": self.P,
            "x_pred": x_pred,
            "P_pred": P_pred,
            "y": y,
            "resid": resid,
            **flags,
        }


# ============================================================
# kalman state initialization
# ============================================================


def init_kalman_state(init_pc: float = 1.0):
    if init_pc <= 0 or not np.isfinite(init_pc):
        raise ValueError("init_pc must be positive")

    state = {
        # Price subspace
        "pc": np.log(float(init_pc)),
        "pt_speed": 0.0,
        "pt_accel": 0.0,
        # Force subspace
        "force_imbalance": 0.0,
        "force_imbalance_trend": 0.0,
        "force_proxy_bias": 0.0,
        # Regime
        "regime_noise_level": 1.0,
    }

    cov = {
        "P_price": np.eye(3),
        "P_force": np.eye(3),
    }

    return state, cov


# ============================================================
# Cycle (mean-reverting) adapter
# ============================================================


def build_cycle_drift_from_mosaic(state, params):
    max_drift = params.get("max_cycle_drift", 3e-3)

    drift = float(state.get("pt_speed", 0.0))
    drift = np.clip(drift, -max_drift, max_drift)

    return drift


class CycleStateAdapter:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def step(
        self, state: Dict[str, Any], P: float, obs_close: float, ctx: Dict[str, Any]
    ):
        x = state["cycle_center"]

        # 異常偏離強度
        obs_noise = abs(state["z_price"])
        z_freeze = self.params.get("z_freeze", None)

        # === JUMP-AWARE 模式切換 ===
        # 小偏離 → 均值回歸 (mean-revert mode)
        # 大偏離 → 直接躍遷 (jump mode, regime shift)
        if z_freeze is not None and obs_noise > z_freeze:
            # 直接跟價格 → 世界觀更新
            return {
                "state": {"cycle_center": obs_close},
                "P": P,  # 保持原先不確定性
                "innov": 0.0,
                "S": np.inf,
                "frozen": True,  # froze = 表示進入 jump mode
            }

        drift = ctx.get("cycle_drift", 0.0)
        x_pred = x * np.exp(drift)

        noise_clip = self.params.get("noise_clip", (-1.0, 5.0))
        obs_clip = self.params.get("obs_clip", (0.0, 5.0))

        cycle_noise = state["regime_noise_level"] - 1.0
        cycle_noise = np.clip(cycle_noise, *noise_clip)
        obs_noise = np.clip(obs_noise, *obs_clip)

        q_scale = self.params.get("q_regime_scale", 1.0)
        r_scale = self.params.get("r_zprice_scale", 1.0)

        Q = self.params["Q_base"] * (1 + q_scale * cycle_noise)
        R = self.params["R_base"] * (1 + r_scale * obs_noise)

        P_pred = P + Q
        innov = obs_close - x_pred
        S = P_pred + R
        K = P_pred / S

        x_new = x_pred + K * innov
        P_new = (1.0 - K) * P_pred

        return {
            "state": {"cycle_center": x_new},
            "P": P_new,
            "innov": innov,
            "S": S,
            "frozen": False,
        }


# ============================================================
# MOSAIC force subspace adapter (3D)
# ============================================================


class MosaicForceAdapter:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def step(
        self,
        state: Dict[str, Any],
        P: np.ndarray,
        obs_force: float,
        ctx: Dict[str, Any],
    ):
        X = np.array(
            [
                state.get("force_imbalance", 0.0),
                state.get("force_imbalance_trend", 0.0),
                state.get("force_proxy_bias", 0.0),
            ]
        )

        rho = float(self.params.get("force_trend_rho", 0.97))
        phi = float(self.params.get("force_proxy_phi", 0.95))

        F = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, rho, 0.0],
                [0.0, 0.0, phi],
            ]
        )

        H = np.array([[1.0, 0.0, 1.0]])

        s = float(np.clip(ctx.get("regime_noise_level", 1.0), 0.2, 5.0))

        Q_base = np.asarray(self.params["Q_force_base"])
        R_base = float(self.params["R_force_base"])

        Q_t = Q_base * (1.0 + self.params.get("q_scale", 0.3) * s)
        R_t = R_base * (1.0 + self.params.get("r_scale", 0.3) * s)

        X_pred = F @ X
        P_pred = F @ P @ F.T + Q_t

        y = np.array([float(obs_force)])
        y_pred = H @ X_pred
        S = (H @ P_pred @ H.T).item() + R_t
        K = (P_pred @ H.T).flatten() / S

        innov = float(y - y_pred)
        X_new = X_pred + K * innov
        P_new = (np.eye(3) - np.outer(K, H.flatten())) @ P_pred

        return {
            "state": {
                "force_imbalance": float(X_new[0]),
                "force_imbalance_trend": float(X_new[1]),
                "force_proxy_bias": float(X_new[2]),
            },
            "P": P_new,
            "innov": innov,
            "S": S,
        }


# ============================================================
# MOSAIC price subspace adapter (3D, one-way force coupling)
# ============================================================


class MosaicPriceAdapter:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def step(
        self,
        state: Dict[str, Any],
        P: np.ndarray,
        obs_close: float,
        ctx: Dict[str, Any],
    ):
        X = np.array(
            [
                state["pc"],
                state["pt_speed"],
                state["pt_accel"],
            ]
        )

        F = np.array(
            [
                [1.0, 1.0, 0.5],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )

        H = np.array([[1.0, 0.0, 0.0]])

        s = float(np.clip(ctx.get("regime_noise_level", 1.0), 0.2, 5.0))
        f = abs(ctx["force_state"]["force_imbalance"])

        Q_base = self.params["Q_price_base"]
        R_base = float(self.params["R_price_base"])

        Q_t = Q_base * (
            1.0
            + self.params.get("q_scale", 0.3) * s
            + self.params.get("q_force_scale", 0.15) * f
        )
        R_t = R_base * (1.0 + self.params.get("r_scale", 0.3) * s)

        X_pred = F @ X
        P_pred = F @ P @ F.T + Q_t

        y = np.array([np.log(max(ZERO, float(obs_close)))])
        y_pred = H @ X_pred
        S = (H @ P_pred @ H.T).item() + R_t
        K = (P_pred @ H.T).flatten() / S

        innov = float(y - y_pred)
        X_new = X_pred + K * innov
        P_new = (np.eye(3) - np.outer(K, H.flatten())) @ P_pred

        return {
            "state": {
                "pc": float(X_new[0]),
                "pt_speed": float(X_new[1]),
                "pt_accel": float(X_new[2]),
            },
            "P": P_new,
            "innov": innov,
            "S": S,
        }


# ============================================================
# Utilities
# ============================================================


def is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def dict_all_finite(d: Dict[str, Any]) -> bool:
    return all(is_finite(v) for v in d.values())


def mat_all_finite(M) -> bool:
    try:
        return np.isfinite(M).all()
    except Exception:
        return False


def step_force(
    state, cov, diagnostics, obs_force, force_model: MosaicForceAdapter, ctx
):
    force_state_prev = {
        "force_imbalance": state.get("force_imbalance", 0.0),
        "force_imbalance_trend": state.get("force_imbalance_trend", 0.0),
        "force_proxy_bias": state.get("force_proxy_bias", 0.0),
    }

    force_out = force_model.step(
        state=force_state_prev,
        P=cov["P_force"],
        obs_force=obs_force,
        ctx=ctx,
    )

    state.update(force_out["state"])
    cov["P_force"] = force_out["P"]

    z_force = force_out["innov"] / np.sqrt(force_out["S"] + 1e-12)
    diagnostics["z_force"] = float(z_force)

    # pass force state forward (one-way)
    ctx["force_state"] = force_out["state"]

    return ctx, z_force


def step_price(
    state, cov, diagnostics, obs_close, price_model: MosaicPriceAdapter, ctx
):
    price_state_prev = {
        "pc": state["pc"],
        "pt_speed": state["pt_speed"],
        "pt_accel": state["pt_accel"],
    }

    price_out = price_model.step(
        state=price_state_prev,
        P=cov["P_price"],
        obs_close=obs_close,
        ctx=ctx,
    )

    state.update(price_out["state"])
    cov["P_price"] = price_out["P"]

    z_price = price_out["innov"] / np.sqrt(price_out["S"] + 1e-12)
    diagnostics["z_price"] = float(z_price)

    return ctx, z_price


def step_regime(state, diagnostics, z_price, z_force, regime_params):
    new_regime, regime_diag = update_regime_noise_level_from_z(
        prev_regime_noise_level=state["regime_noise_level"],
        z_price=z_price,
        z_force=z_force,
        params=regime_params,
    )

    state["regime_noise_level"] = new_regime
    diagnostics.update(regime_diag)


def step_cycle(state, cov, diagnostics, obs_close, cycle_model: CycleStateAdapter, ctx):
    cycle_state_prev = {
        "cycle_center": state.get("cycle_center", obs_close),
        "regime_noise_level": state["regime_noise_level"],
        "z_price": diagnostics["z_price"],
    }
    ctx["cycle_drift"] = build_cycle_drift_from_mosaic(state, cycle_model.params)

    cycle_out = cycle_model.step(
        state=cycle_state_prev,
        P=cov.get("P_cycle", 1.0),
        obs_close=obs_close,
        ctx=ctx,
    )

    state.update(cycle_out["state"])
    cov["P_cycle"] = cycle_out["P"]

    z_cycle = cycle_out["innov"] / np.sqrt(cycle_out["S"] + 1e-12)
    diagnostics["z_cycle"] = float(z_cycle)
    return ctx


def mosaic_step(
    state: Dict[str, Any],
    cov: Dict[str, Any],
    obs_close: float,
    obs_force: float,
    force_model: MosaicForceAdapter,
    price_model: MosaicPriceAdapter,
    cycle_model: CycleStateAdapter,
    regime_params: Dict[str, Any],
):
    """
    Fused MOSAIC step:
    Force → Price → Regime → Cycle
    """

    diagnostics = {}

    # =========================================================
    # Context (shared)
    # =========================================================
    ctx = {
        "regime_noise_level": float(state["regime_noise_level"]),
    }

    # =========================================================
    # FORCE subspace
    # =========================================================
    ctx, z_force = step_force(state, cov, diagnostics, obs_force, force_model, ctx)

    # =========================================================
    # PRICE subspace
    # =========================================================
    ctx, z_price = step_price(state, cov, diagnostics, obs_close, price_model, ctx)

    # =========================================================
    # REGIME update (from normalized surprise)
    # =========================================================
    step_regime(state, diagnostics, z_price, z_force, regime_params)

    # =========================================================
    # CYCLE (mean-reverting structural center)
    # =========================================================
    ctx = step_cycle(state, cov, diagnostics, obs_close, cycle_model, ctx)

    # =========================================================
    # EXPORT snapshot
    # =========================================================
    diagnostics.update(
        {
            "pc": state["pc"],
            "pt_speed": state["pt_speed"],
            "pt_accel": state["pt_accel"],
            "force_imbalance": state["force_imbalance"],
            "force_imbalance_trend": state["force_imbalance_trend"],
            "regime_noise_level": state["regime_noise_level"],
            "cycle_center": state["cycle_center"],
        }
    )

    return state, cov, diagnostics


def update_regime_noise_level_from_z(
    prev_regime_noise_level: float,
    z_price: float,
    z_force: float,
    params: Dict[str, Any],
):
    """
    Use normalized innovations (z = innov / sqrt(S))
    to adapt regime noise level.
    """

    z_price = abs(float(z_price))
    z_force = abs(float(z_force))

    w_price = params.get("w_price", 0.6)
    w_force = params.get("w_force", 0.4)
    z_mix = w_price * z_price + w_force * z_force

    base = params.get("base", 1.0)
    gain = params.get("gain", 0.3)

    # only react when surprise > 1σ (soft)
    target = base + gain * np.log1p(max(0.0, z_mix - 1.0))

    beta = params.get("regime_smooth", 0.15)
    new_regime = (1 - beta) * float(prev_regime_noise_level) + beta * float(target)

    lo, hi = params.get("clip", (0.2, 5.0))
    new_regime = float(np.clip(new_regime, lo, hi))

    return new_regime, {
        "z_mix": float(z_mix),
        "target_regime": float(target),
        "regime_noise_level_new": new_regime,
    }


def build_kalman_params():
    params = {
        "price": {
            "Q_price_base": np.diag([1e-3, 1e-4, 1e-5]),
            "R_price_base": 1e-1,
            "q_scale": 0.3,
            "r_scale": 0.3,
            "q_force_scale": 0.15,
            "force_scale_clip": 5.0,
            "accel_inject": 0.0,  # keep honest by default
        },
        "force": {
            "Q_force_base": np.diag([3e-5, 1e-5, 1e-5]),
            "R_force_base": 1e-1,
            "q_scale": 0.3,
            "r_scale": 0.3,
            "force_trend_rho": 0.97,
            "force_proxy_phi": 0.90,
        },
        "cycle": {
            "Q_base": 1e-4,
            "R_base": 1e-2,
            "q_regime_scale": 0.8,  # regime_noise → Q
            "r_zprice_scale": 1.0,  # |z_price| → R
            "noise_clip": (-0.5, 5.0),
            "obs_clip": (0.0, 5.0),
            "z_freeze": 3.0,  # 超過就 freeze cycle
        },
    }

    regime_params = {
        "w_price": 0.6,
        "w_force": 0.4,
        "base": 1.0,
        "gain": 0.8,
        "regime_smooth": 0.05,
        "clip": (0.2, 5.0),
    }

    return params, regime_params


def prepare_mosaic_input(df):
    """
    df 至少包含: Open, High, Low, Close, Vol
    ForceProxy v2 (三因子融合):
      - CoreForce:   ΔClose * Vol
      - Efficiency:  ΔClose / max(Vol, eps)
      - Cumulative:  OBV-like sign(ΔClose) * Vol 累积
    """

    df = df.copy()
    eps = 1e-12
    w = 200  # 可调

    # ΔClose: 推动方向 + 粗量级
    d_close = df["Close"].diff().fillna(0.0)

    # Core Force: 方向 × 资金量
    core_force = d_close * df["Vol"]

    # Efficiency: 性价比 = 推动 / 成本
    eff = d_close / (df["Vol"].replace(0, eps))

    # Cumulative: 趋势累积偏压
    thr = df["Close"].pct_change().abs().rolling(w).median() * 0.5
    cvd = (
        (np.sign(d_close) * df["Vol"] * (abs(d_close) > thr))
        .rolling(w, min_periods=1)
        .sum()
    )

    # Rolling Normalize (online-ish)

    def z_roll(x):
        m = x.rolling(w).mean()
        s = x.rolling(w).std().replace(0, 1.0)
        return ((x - m) / s).fillna(0.0)

    df["force_core"] = z_roll(core_force)
    df["force_eff"] = z_roll(eff)
    df["force_cum"] = z_roll(cvd)

    # ✨ Final Force Proxy 合成（简单平均，未来可加权）
    df["force_proxy"] = (df["force_core"] + df["force_eff"] + df["force_cum"]) / 3.0
    return df


import numpy as np
import pandas as pd


def update_liquidity_wall(df):
    """
    Symmetric liquidity wall (research version, pandas-first)

    Goal:
    - Observe whether liquidity exhaustion boundary exists
    - NOT trading logic

    Key ideas:
    - Wall anchored at c_center
    - Width from price state uncertainty (P_price), not price movement
    - Wall must be slower than price
    """

    # ====== tunable constants (edit freely) ======
    K_STATE = 2.0  # how many sigma = natural reachable radius
    NOISE_MIN = 0.8
    NOISE_MAX = 1.5
    ATR_FLOOR = 0.8  # ATR is safety fuse only
    WALL_ALPHA = 0.05  # wall inertia (must be slow)

    # ====== ensure columns exist ======
    for col in ["c_width", "c_upper", "c_lower", "z_dist"]:
        if col not in df.columns:
            df[col] = np.nan

    # ====== decide which rows need update ======
    # rule: if c_width is NaN, we consider this row "not computed yet"
    need_update = df["c_width"].isna()

    if not need_update.any():
        return df  # nothing to do

    # ====== core inputs ======
    c_center = df["c_center"]
    P_price = df["m_p_price"]
    noise = df["m_regime_noise_level"].clip(NOISE_MIN, NOISE_MAX)
    atr = df["ATR"]

    # ====== step 1: base width from state uncertainty ======
    # prior, not posterior
    base_width = K_STATE * np.sqrt(P_price.clip(lower=1e-12))

    # ====== step 2: regime modulation (bounded, gentle) ======
    width_structural = base_width * noise

    # ====== step 3: ATR as safety floor ======
    width_raw = np.maximum(width_structural, ATR_FLOOR * atr)

    # ====== step 4: wall inertia (EWMA, but ONLY where needed) ======
    # We cannot pure-vectorize this part because it is recursive.
    # But we keep it minimal and readable.

    c_width = df["c_width"].copy()

    for i in df.index[need_update]:
        if i == df.index[0]:
            c_width.at[i] = width_raw.at[i]
            continue

        prev = c_width.at[i - 1]
        if np.isfinite(prev):
            c_width.at[i] = (1 - WALL_ALPHA) * prev + WALL_ALPHA * width_raw.at[i]
        else:
            c_width.at[i] = width_raw.at[i]

    df["c_width"] = c_width

    # ====== step 5: symmetric wall ======
    df.loc[need_update, "c_upper"] = c_center + df["c_width"]
    df.loc[need_update, "c_lower"] = c_center - df["c_width"]

    # ====== step 6: normalized distance (this is what we actually look at) ======
    df.loc[need_update, "z_dist"] = (df["Close"] - c_center) / df["c_width"]

    # force / speed 的絕對值（避免正負干擾）
    df.loc[need_update, "force_abs"] = df["m_force_trend"].abs()
    df.loc[need_update, "speed_abs"] = df["m_pt_speed"].abs()

    # 市場阻抗（你前面其實已經在用這個概念）
    df.loc[need_update, "impedance"] = df["force_abs"] / (df["speed_abs"] + 1e-6)

    Z_CORE = 0.5  # 靠近 c_center
    Z_EXHAUST = 1.5  # 接近邊界

    FORCE_Q = df["force_abs"].quantile(0.7)
    SPEED_Q = df["speed_abs"].quantile(0.7)
    IMPEDANCE_Q = df["impedance"].quantile(0.7)

    df.loc[need_update, "market_phase"] = 0

    mask_inertia_core = (
        (df["z_dist"].abs() < Z_CORE)
        & (df["force_abs"] > FORCE_Q)
        & (df["speed_abs"] < SPEED_Q)
    )

    df.loc[mask_inertia_core, "market_phase"] = 1

    mask_exhaustion_push = (
        (df["z_dist"].abs() >= Z_EXHAUST)
        & (df["force_abs"] > FORCE_Q)
        & (df["impedance"] > IMPEDANCE_Q)
    )

    df.loc[mask_exhaustion_push, "market_phase"] = -1

    mask_vacuum_break = (
        (df["z_dist"].abs() >= Z_EXHAUST)
        & (df["speed_abs"] > SPEED_Q)
        & (df["force_abs"] <= FORCE_Q)
    )

    df.loc[mask_vacuum_break, "market_phase"] = 2
    return df


if __name__ == "__main__":
    # =========================================================
    # Simple sanity test for MOSAIC
    # =========================================================

    np.random.seed(42)

    # ---------- build params & models ----------
    params, regime_params = build_mosaic_params()

    force_model = MosaicForceAdapter(params["force"])
    price_model = MosaicPriceAdapter(params["price"])

    cycle_model = CycleStateAdapter(
        params={
            "Q_base": 0.01,
            "R_base": 0.5,
        }
    )

    # ---------- init state ----------
    state, cov = init_mosaic_state(init_pc=100.0)

    # give cycle an initial center
    state["cycle_center"] = 100.0
    cov["P_cycle"] = 1.0

    # ---------- generate fake market ----------
    T = 300

    price = 100.0
    force = 0.0

    print("t | price | force | z_force | z_price | z_cycle | regime")
    print("-" * 70)

    for t in range(T):
        # ---- synthetic force process ----
        if 100 < t < 160:
            force += np.random.normal(0.2, 0.1)  # stress / imbalance
        else:
            force += np.random.normal(0.0, 0.05)

        # ---- synthetic price process ----
        price += 0.05 * force + np.random.normal(0.0, 0.3)

        # ---- step mosaic ----
        state, cov, diag = mosaic_step(
            state=state,
            cov=cov,
            obs_close=price,
            obs_force=force,
            force_model=force_model,
            price_model=price_model,
            cycle_model=cycle_model,
            regime_params=regime_params,
        )

        # ---- log occasionally ----
        if t % 10 == 0:
            print(
                f"{t:3d} | "
                f"{price:6.2f} | "
                f"{force:6.2f} | "
                f"{diag['z_force']:7.3f} | "
                f"{diag['z_price']:7.3f} | "
                f"{diag['z_cycle']:7.3f} | "
                f"{state['regime_noise_level']:6.3f}"
            )

    print("\nFinal state snapshot:")
    for k, v in state.items():
        print(f"{k:>24s} : {v}")
