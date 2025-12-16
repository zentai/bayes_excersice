import numpy as np
from numpy.linalg import inv
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore, t


# ======================================================
# 1) Synthetic World
# ======================================================


def generate_synthetic_force_v2(
    T=2000,
    rho=0.98,
    trend_sigma=0.01,
    force_sigma=0.05,
    seed=42,
):
    """
    Structural force world:
      force has persistence + slow trend
    """
    rng = np.random.default_rng(seed)

    true_trend = np.zeros(T)
    true_force = np.zeros(T)

    for t in range(1, T):
        true_trend[t] = true_trend[t - 1] + rng.normal(0, trend_sigma)
        true_force[t] = (
            rho * true_force[t - 1] + true_trend[t] + rng.normal(0, force_sigma)
        )

    return {
        "true_force": true_force,
        "true_trend": true_trend,
    }


# ======================================================
# 2) Observation Layer
# ======================================================


def generate_force_observation(
    true_force,
    noise_scale=1.5,
    heavy_tail=True,
    seed=123,
):
    """
    pseudo_delta style observation:
      obs = true_force + microstructure noise
    """
    rng = np.random.default_rng(seed)

    if heavy_tail:
        noise = t(df=3).rvs(len(true_force), random_state=rng) * noise_scale
    else:
        noise = rng.normal(0, noise_scale, size=len(true_force))

    return true_force + noise


# ======================================================
# 3) Force Pipeline (Kalman Subspace)
# ======================================================


def run_force_pipeline(obs_force):
    """
    Simple 2D force Kalman:
      x = [force_imbalance, force_trend]
    """
    T = len(obs_force)

    F = np.array([[1.0, 1.0], [0.0, 0.98]])
    H = np.array([[1.0, 0.0]])

    Q = np.diag([1e-4, 1e-5])
    R = np.array([[1.0]])

    x = np.zeros(2)
    P = np.eye(2)

    est_force = np.zeros(T)
    est_trend = np.zeros(T)
    innovations = np.zeros(T)

    I = np.eye(2)

    for t in range(T):
        # predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # update
        y = np.array([obs_force[t]])
        y_pred = H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ inv(S)

        innov = (y - y_pred).item()
        x = x_pred + (K.flatten() * innov)
        P = (I - K @ H) @ P_pred

        est_force[t] = x[0]
        est_trend[t] = x[1]
        innovations[t] = innov

    return est_force, est_trend, innovations


# ======================================================
# 4) Audits (Correct Force Tests)
# ======================================================


def audit_force_pipeline(true_force, obs_force, est_force, innovations):
    report = {}

    # --- A) amplitude sanity ---
    z = zscore(innovations)
    extreme_ratio = np.mean(np.abs(z) > 8)
    report["extreme_ratio"] = extreme_ratio

    # --- B) information reduction ---
    acf_raw = acf(obs_force, nlags=1, fft=False)[1]
    acf_innov = acf(innovations, nlags=1, fft=False)[1]
    report["acf_raw_lag1"] = acf_raw
    report["acf_innov_lag1"] = acf_innov
    report["acf_reduction_ratio"] = acf_innov / acf_raw

    # --- C) no oracle test ---
    corr_now = np.corrcoef(innovations, true_force)[0, 1]
    corr_future = np.corrcoef(innovations[:-1], true_force[1:])[0, 1]
    report["corr_now"] = corr_now
    report["corr_future"] = corr_future

    # --- D) structural alignment ---
    corr_obs = np.corrcoef(obs_force, true_force)[0, 1]
    corr_est = np.corrcoef(est_force, true_force)[0, 1]
    report["corr_obs_true"] = corr_obs
    report["corr_est_true"] = corr_est

    return report


# ======================================================
# 5) Main Runner
# ======================================================


def main():
    # 1) synthetic world
    world = generate_synthetic_force_v2()
    true_force = world["true_force"]

    # 2) observation
    obs_force = generate_force_observation(
        true_force,
        noise_scale=1.5,
        heavy_tail=True,
    )

    # 3) run pipeline
    est_force, est_trend, innovations = run_force_pipeline(obs_force)

    # 4) audits
    report = audit_force_pipeline(
        true_force=true_force,
        obs_force=obs_force,
        est_force=est_force,
        innovations=innovations,
    )

    print("\n========== FORCE PIPELINE AUDIT ==========")
    for k, v in report.items():
        print(f"{k:25s}: {v}")

    print("\n=== Interpretation Guide ===")
    print("extreme_ratio < 0.01                → 数值稳定")
    print("acf_innov < acf_raw                → 吸收结构")
    print("|corr_future| <= |corr_now|        → 无偷看未来")
    print("corr_est_true > corr_obs_true      → 提取 latent force")


if __name__ == "__main__":
    main()
