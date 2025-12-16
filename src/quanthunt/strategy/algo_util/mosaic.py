import numpy as np

"""
MOSAIC State Model
------------------
MOSAIC = Market-Oriented State And Imbalance Core

This module implements the Force Subspace of the MOSAIC model.

Design principles:
- Force state represents latent orderflow imbalance
- No price return is used as observation
- Fully recursive (no rolling window)
- Compatible with BOCPD / HMM as downstream consumers
"""


def init_mosaic_state(init_pc=0.0):
    state = {
        # Price subspace
        "pc": float(init_pc),
        "pt_speed": 0.0,
        "pt_accel": 0.0,
        # Force subspace
        "force_imbalance": 0.0,
        "force_imbalance_trend": 0.0,
        "force_proxy_bias": 0.0,
        # Environment / regime
        "regime_noise_level": 1.0,
    }

    cov = {
        "P_price": np.eye(3) * 1.0,
        "P_force": np.eye(3) * 1.0,
    }
    return state, cov


# ------------------------------
# Subspace steps (no circularity)
# ------------------------------


def mosaic_force_kalman_step(
    prev_force_state, prev_force_cov, obs_force, regime_noise_level, params
):
    """
    MOSAIC = Market-Oriented State And Imbalance Core

    Force Subspace (3D, colored measurement handled):
      x = [force_imbalance, force_imbalance_trend, force_proxy_bias]
      y = obs_force (e.g., pseudo_delta z)

    Observation:
      y = force_imbalance + force_proxy_bias + e_t
      e_t ~ N(0, R_t)

    force_proxy_bias is AR(1) to absorb proxy autocorrelation.
    """

    if prev_force_cov.shape != (3, 3):
        raise ValueError(f"P_force must be (3,3), got {prev_force_cov.shape}")

    # --- state vector ---
    x = np.array(
        [
            prev_force_state.get("force_imbalance", 0.0),
            prev_force_state.get("force_imbalance_trend", 0.0),
            prev_force_state.get("force_proxy_bias", 0.0),
        ],
        dtype=float,
    )
    P = prev_force_cov
    I = np.eye(3)

    # --- dynamics ---
    rho = float(params.get("force_trend_rho", 0.97))
    phi = float(params.get("force_proxy_phi", 0.90))  # <- new: proxy bias persistence

    F = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, rho, 0.0],
            [0.0, 0.0, phi],
        ],
        dtype=float,
    )

    # observe force_imbalance + proxy_bias
    H = np.array([[1.0, 0.0, 1.0]], dtype=float)

    # --- noise ---
    Q_base = np.asarray(params["Q_force_base"], dtype=float)  # expect 3x3 now
    R_base = float(params["R_force_base"])

    s = float(np.clip(float(regime_noise_level), 0.2, 5.0))

    q_scale = float(params.get("q_scale", 0.3))
    r_scale = float(params.get("r_scale", 0.3))

    Q_t = Q_base * (1.0 + q_scale * s)
    R_t = R_base * (1.0 + r_scale * s)

    # --- predict ---
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q_t

    # --- update ---
    y = np.array([float(obs_force)], dtype=float)
    y_pred = H @ x_pred  # (1,)

    S = (H @ P_pred @ H.T).item() + R_t  # scalar
    K = (P_pred @ H.T).flatten() / S  # (3,)

    innov = float((y - y_pred).item())
    x_new = x_pred + K * innov
    P_new = (I - np.outer(K, H.flatten())) @ P_pred

    new_force_state = {
        "force_imbalance": float(x_new[0]),
        "force_imbalance_trend": float(x_new[1]),
        "force_proxy_bias": float(x_new[2]),
    }
    return new_force_state, P_new, innov, float(S)


def mosaic_price_kalman_step(
    prev_price_state, prev_price_cov, obs_close, regime_noise_level, force_state, params
):
    """
    MOSAIC = Market-Oriented State And Imbalance Core
    Price Subspace:
      x = [pc, pt_speed, pt_accel]
      y = Close
    Important: Force → Price is one-way only.
    """
    x = np.array(
        [
            prev_price_state["pc"],
            prev_price_state["pt_speed"],
            prev_price_state["pt_accel"],
        ],
        dtype=float,
    )
    P = prev_price_cov

    F = np.array(
        [
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    H = np.array([[1.0, 0.0, 0.0]])
    I = np.eye(3)

    Q_base = params["Q_price_base"]  # 3x3
    R_base = float(params["R_price_base"])

    # --- environment scaling ---
    s = float(regime_noise_level)
    s = np.clip(s, 0.2, 5.0)

    # --- force coupling (one-way) ---
    # Option A: modulate process noise (safe, no bias injection)
    f = float(force_state["force_imbalance"])
    f_scale = np.clip(abs(f), 0.0, params.get("force_scale_clip", 5.0))

    q_scale = params.get("q_scale", 0.3)
    r_scale = params.get("r_scale", 0.3)
    q_force_scale = params.get("q_force_scale", 0.15)  # how much force inflates Q

    Q_t = Q_base * (1.0 + q_scale * s + q_force_scale * f_scale)
    R_t = R_base * (1.0 + r_scale * s)

    # Option B (optional): inject accel drift from force (still one-way, but introduces bias)
    # Keep OFF by default for "honest" filtering.
    accel_inject = params.get("accel_inject", 0.0)  # 0.0 means disabled
    if accel_inject != 0.0:
        x[2] += accel_inject * f  # nudges accel prior (not using future info)

    # predict
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q_t

    # update
    y = np.array([float(obs_close)])
    y_pred = H @ x_pred
    S = H @ P_pred @ H.T + R_t
    K = (P_pred @ H.T) / S

    innov = (y - y_pred).item()
    x_new = x_pred + (K.flatten() * innov)
    P_new = (I - K @ H) @ P_pred

    new_price_state = {"pc": x_new[0], "pt_speed": x_new[1], "pt_accel": x_new[2]}
    return new_price_state, P_new, float(innov), float(S.item())


# ------------------------------
# MOSAIC.step() (fused, no loops)
# ------------------------------
import numpy as np


def _is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _dict_all_finite(d: dict) -> bool:
    for v in d.values():
        if not _is_finite(v):
            return False
    return True


def _mat_all_finite(M) -> bool:
    try:
        return np.isfinite(M).all()
    except Exception:
        return False


def mosaic_step(
    MOSAIC_State,
    MOSAIC_Cov,
    obs_close,
    obs_force_proxy,
    params,
):
    """
    MOSAIC = Market-Oriented State And Imbalance Core

    Safe version (minimal invasive):
    - guard NaN/inf in obs
    - guard NaN/inf in updates
    - guard innovation variance S (must be finite and > 0)
    - rollback if anything goes wrong
    """
    # ===== rollback snapshot =====
    _state_prev = dict(MOSAIC_State)
    _cov_prev = {
        "P_force": MOSAIC_Cov["P_force"].copy(),
        "P_price": MOSAIC_Cov["P_price"].copy(),
    }

    diagnostics = {"safe_skip": False, "safe_reason": None}

    # --- unpack ---
    regime_noise_level = float(MOSAIC_State["regime_noise_level"])

    # ===== 0) obs guard =====
    if not _is_finite(obs_close) or not _is_finite(obs_force_proxy):
        diagnostics.update(
            {
                "safe_skip": True,
                "safe_reason": "obs_not_finite",
            }
        )
        return MOSAIC_State, MOSAIC_Cov, diagnostics

    try:
        # 1) Force update (uses only obs_force_proxy + regime)
        force_state_prev = {
            "force_imbalance": MOSAIC_State["force_imbalance"],
            "force_imbalance_trend": MOSAIC_State["force_imbalance_trend"],
            "force_proxy_bias": MOSAIC_State.get("force_proxy_bias", 0.0),
        }

        new_force_state, P_force_new, force_innov, force_S = mosaic_force_kalman_step(
            prev_force_state=force_state_prev,
            prev_force_cov=MOSAIC_Cov["P_force"],
            obs_force=obs_force_proxy,
            regime_noise_level=regime_noise_level,
            params=params["force"],
        )

        # --- force guards ---
        if (not _dict_all_finite(new_force_state)) or (
            not _mat_all_finite(P_force_new)
        ):
            raise FloatingPointError("force_nan")
        if (not _is_finite(force_S)) or (force_S <= 0.0):
            raise FloatingPointError("force_S_invalid")
        if not _is_finite(force_innov):
            raise FloatingPointError("force_innov_nan")

        # 2) Price update (uses obs_close + one-way coupling)
        price_state_prev = {
            "pc": MOSAIC_State["pc"],
            "pt_speed": MOSAIC_State["pt_speed"],
            "pt_accel": MOSAIC_State["pt_accel"],
        }

        new_price_state, P_price_new, price_innov, price_S = mosaic_price_kalman_step(
            prev_price_state=price_state_prev,
            prev_price_cov=MOSAIC_Cov["P_price"],
            obs_close=obs_close,
            regime_noise_level=regime_noise_level,
            force_state=new_force_state,  # one-way coupling
            params=params["price"],
        )

        # --- price guards ---
        if (not _dict_all_finite(new_price_state)) or (
            not _mat_all_finite(P_price_new)
        ):
            raise FloatingPointError("price_nan")
        if (not _is_finite(price_S)) or (price_S <= 0.0):
            raise FloatingPointError("price_S_invalid")
        if not _is_finite(price_innov):
            raise FloatingPointError("price_innov_nan")

        # 3) Commit update
        MOSAIC_State.update(new_force_state)
        MOSAIC_State.update(new_price_state)
        MOSAIC_Cov["P_force"] = P_force_new
        MOSAIC_Cov["P_price"] = P_price_new

        # ===== final state/cov sanity =====
        # (避免你那种 dump 第一笔正常，后面全 NaN 的情况)
        required_keys = [
            "pc",
            "pt_speed",
            "pt_accel",
            "force_imbalance",
            "force_imbalance_trend",
            "regime_noise_level",
        ]
        for k in required_keys:
            if not _is_finite(MOSAIC_State.get(k, np.nan)):
                raise FloatingPointError(f"final_state_nan:{k}")
        if not _mat_all_finite(MOSAIC_Cov["P_force"]) or not _mat_all_finite(
            MOSAIC_Cov["P_price"]
        ):
            raise FloatingPointError("final_cov_nan")

        # diagnostics (keep your original)
        eps = 1e-12
        z_force = force_innov / np.sqrt(force_S + eps)
        z_price = price_innov / np.sqrt(price_S + eps)

        diagnostics.update(
            {
                "force_innov": float(force_innov),
                "force_S": float(force_S),
                "z_force": float(z_force),
                "price_innov": float(price_innov),
                "price_S": float(price_S),
                "z_price": float(z_price),
                "pc": MOSAIC_State["pc"],
                "pt_speed": MOSAIC_State["pt_speed"],
                "pt_accel": MOSAIC_State["pt_accel"],
                "force_imbalance": MOSAIC_State["force_imbalance"],
                "force_imbalance_trend": MOSAIC_State["force_imbalance_trend"],
                "force_proxy_bias": MOSAIC_State.get("force_proxy_bias", 0.0),
                "regime_noise_level": MOSAIC_State["regime_noise_level"],
            }
        )

        return MOSAIC_State, MOSAIC_Cov, diagnostics

    except Exception as e:
        # ===== rollback =====
        MOSAIC_State.clear()
        MOSAIC_State.update(_state_prev)

        MOSAIC_Cov["P_force"] = _cov_prev["P_force"]
        MOSAIC_Cov["P_price"] = _cov_prev["P_price"]

        diagnostics.update(
            {
                "safe_skip": True,
                "safe_reason": str(e),
            }
        )
        return MOSAIC_State, MOSAIC_Cov, diagnostics


def update_regime_noise_level_from_z(prev_regime_noise_level, z_price, z_force, params):
    """
    Use normalized innovations z = innov/sqrt(S) as the regime signal.
    If model is roughly correct, z has comparable scale across subspaces.
    """
    z_price = abs(float(z_price))
    z_force = abs(float(z_force))

    w_price = params.get("w_price", 0.6)
    w_force = params.get("w_force", 0.4)
    z_mix = w_price * z_price + w_force * z_force

    base = params.get("base", 1.0)
    gain = params.get("gain", 0.8)

    # only react when surprise > 1 sigma (soft)
    target = base + gain * np.log1p(max(0.0, z_mix - 1.0))

    beta = params.get("regime_smooth", 0.05)
    new_regime = (1 - beta) * float(prev_regime_noise_level) + beta * float(target)

    lo, hi = params.get("clip", (0.2, 5.0))
    new_regime = float(np.clip(new_regime, lo, hi))

    return new_regime, {
        "z_mix": float(z_mix),
        "target_regime": float(target),
        "regime_noise_level_new": new_regime,
    }
