import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


def generate_synthetic_price_state(
    T=2000,
    rho_speed=0.98,  # speed damping
    rho_accel=0.95,  # accel damping
    sigma_accel=0.02,
    sigma_speed=0.01,
    sigma_level=0.0,  # usually 0 for pure kinematics
    regime_shift_prob=0.002,
    seed=7,
):
    """
    True latent state:
      pc, pt_speed, pt_accel
    """
    rng = np.random.default_rng(seed)

    pc = np.zeros(T)
    pt_speed = np.zeros(T)
    pt_accel = np.zeros(T)

    for t in range(1, T):
        # occasional regime shift (accel kick)
        if rng.random() < regime_shift_prob:
            pt_accel[t] = rng.normal(0, 0.15)
        else:
            pt_accel[t] = rho_accel * pt_accel[t - 1] + rng.normal(0, sigma_accel)

        pt_speed[t] = (
            rho_speed * pt_speed[t - 1] + pt_accel[t] + rng.normal(0, sigma_speed)
        )
        pc[t] = pc[t - 1] + pt_speed[t] + 0.5 * pt_accel[t] + rng.normal(0, sigma_level)

    return pc, pt_speed, pt_accel


def generate_price_observation_close(
    true_pc,
    obs_noise=0.5,
    heavy_tail=True,
    seed=9,
):
    rng = np.random.default_rng(seed)
    if heavy_tail:
        noise = rng.standard_t(df=3, size=len(true_pc)) * obs_noise
    else:
        noise = rng.normal(0, obs_noise, size=len(true_pc))

    close = true_pc + noise
    return close


def mosaic_price_kalman_step(
    prev_price_state,
    prev_price_cov,
    obs_close,
    regime_noise_level,
    params,
):
    """
    MOSAIC State Model
    ------------------
    MOSAIC = Market-Oriented State And Imbalance Core

    Price Subspace:
      x = [pc, pt_speed, pt_accel]
      y = Close
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

    # 3rd-order kinematics
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
    R_base = params["R_price_base"]  # scalar

    # environment-driven noise scaling (keep monotonic + clipped)
    s = float(regime_noise_level)
    s = np.clip(s, 0.2, 5.0)

    Q_t = Q_base * (1.0 + params["q_scale"] * s)
    R_t = R_base * (1.0 + params["r_scale"] * s)

    # predict
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q_t

    # update
    y = np.array([obs_close])
    y_pred = H @ x_pred

    S = H @ P_pred @ H.T + R_t
    K = P_pred @ H.T / S

    innov = (y - y_pred).item()
    x_new = x_pred + (K.flatten() * innov)
    P_new = (I - K @ H) @ P_pred

    price_state = {"pc": x_new[0], "pt_speed": x_new[1], "pt_accel": x_new[2]}
    return price_state, P_new, innov


def autocorr(x, lag=1):
    x = np.asarray(x)
    if lag >= len(x) or lag <= 0:
        return np.nan
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


def whiteness_report(innov, true_series=None, name=""):
    report = {}
    report["acf_lag1"] = autocorr(innov, 1)
    report["acf_lag2"] = autocorr(innov, 2)
    lb = acorr_ljungbox(innov, lags=[10], return_df=True)
    report["ljung_box_p"] = lb["lb_pvalue"].iloc[0]
    if true_series is not None:
        report["corr_innov_true"] = np.corrcoef(innov, true_series)[0, 1]

    print(f"\n--- Whiteness Report {name} ---")
    for k, v in report.items():
        print(f"{k:30s}: {v: .4f}")
    return report


def run_price_pipeline(T=2000):
    # 1) true world
    true_pc, true_speed, true_accel = generate_synthetic_price_state(T=T)

    # 2) observe Close
    close = generate_price_observation_close(true_pc, obs_noise=0.8, heavy_tail=True)

    # 3) kalman params
    price_params = {
        "Q_price_base": np.diag([1e-5, 1e-4, 1e-3]),
        "R_price_base": 0.8,  # roughly matches obs_noise scale
        "q_scale": 0.3,
        "r_scale": 0.3,
    }

    state = {"pc": close[0], "pt_speed": 0.0, "pt_accel": 0.0}
    P = np.eye(3)

    est_pc = np.zeros(T)
    est_speed = np.zeros(T)
    est_accel = np.zeros(T)
    innovs = np.zeros(T)

    regime_noise_level = 1.0  # synthetic: keep constant for first pass

    for t in range(T):
        state, P, innov = mosaic_price_kalman_step(
            prev_price_state=state,
            prev_price_cov=P,
            obs_close=close[t],
            regime_noise_level=regime_noise_level,
            params=price_params,
        )
        est_pc[t] = state["pc"]
        est_speed[t] = state["pt_speed"]
        est_accel[t] = state["pt_accel"]
        innovs[t] = innov

    # 4) reports
    # raw proxy innovation = Close - true_pc  (this is the pure observation noise in our synthetic world)
    raw_resid = close - true_pc
    whiteness_report(
        raw_resid, true_series=true_pc, name="Raw Close Residual (Close-true_pc)"
    )
    whiteness_report(innovs, true_series=true_pc, name="Price Innovation")

    return {
        "true_pc": true_pc,
        "true_speed": true_speed,
        "true_accel": true_accel,
        "close": close,
        "est_pc": est_pc,
        "est_speed": est_speed,
        "est_accel": est_accel,
        "innov": innovs,
    }


if __name__ == "__main__":
    out = run_price_pipeline()
