import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from whitenoise_market import build_synthetic_market_ohlcv
from mosaic import init_mosaic_state, mosaic_step, update_regime_noise_level_from_z


def whiteness_report(z, lags=20):
    z = np.asarray(z)
    z = z[~np.isnan(z)]

    acf_vals = acf(z, nlags=2, fft=True)
    lb = acorr_ljungbox(z, lags=[lags], return_df=True)

    return {
        "acf_lag1": float(acf_vals[1]) if len(acf_vals) > 1 else np.nan,
        "acf_lag2": float(acf_vals[2]) if len(acf_vals) > 2 else np.nan,
        "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
    }


def run_gate_suite(z_price, z_force, regime_series, returns):
    report = {}

    # --- Gate A: extreme values ---
    report["z_price_extreme_ratio"] = float(np.mean(np.abs(z_price) > 10))
    report["z_force_extreme_ratio"] = float(np.mean(np.abs(z_force) > 10))

    report["gate_A_pass"] = (
        report["z_price_extreme_ratio"] < 0.01
        and report["z_force_extreme_ratio"] < 0.01
    )

    # --- Gate B: whiteness ---
    report["whiteness_price"] = whiteness_report(z_price)
    report["whiteness_force"] = whiteness_report(z_force)

    acf_lag1 = report["whiteness_force"].get("acf_lag1")
    acf_lag2 = report["whiteness_force"].get("acf_lag2")

    report["gate_B_pass"] = abs(acf_lag1) < 0.6 and abs(acf_lag2) < 0.5

    # --- Gate C: causal sanity ---
    ret = pd.Series(returns)
    zf = pd.Series(z_force)

    report["corr_force_current_ret"] = float(zf.corr(ret))
    report["corr_force_future_ret"] = float(zf.corr(ret.shift(-1)))

    # soft expectation
    report["gate_C_hint"] = abs(report["corr_force_future_ret"]) > abs(
        report["corr_force_current_ret"]
    )

    # --- Gate D: regime consistency ---
    abs_z = np.abs(z_price)
    regime = pd.Series(regime_series)

    high_regime = abs_z[regime > regime.median()].mean()
    low_regime = abs_z[regime <= regime.median()].mean()

    report["z_high_regime_mean"] = float(high_regime)
    report["z_low_regime_mean"] = float(low_regime)
    report["gate_D_hint"] = high_regime > low_regime

    return report


def run_mosaic_runner(df, mosaic_init_fn, mosaic_step_fn, params, regime_params):
    """
    df must contain:
      - Close
      - force_proxy (scalar)
    """
    state, cov = mosaic_init_fn(init_pc=df["Close"].iloc[0])

    z_price_series = []
    z_force_series = []
    regime_series = []
    returns = df["Close"].pct_change().fillna(0.0).values

    for t in range(len(df)):
        state, cov, diag = mosaic_step_fn(
            MOSAIC_State=state,
            MOSAIC_Cov=cov,
            obs_close=df["Close"].iloc[t],
            obs_force_proxy=df["force_proxy"].iloc[t],
            params=params,
        )

        # update regime
        state["regime_noise_level"], _ = update_regime_noise_level_from_z(
            prev_regime_noise_level=state["regime_noise_level"],
            z_price=diag["z_price"],
            z_force=diag["z_force"],
            params=regime_params,
        )

        z_price_series.append(diag["z_price"])
        z_force_series.append(diag["z_force"])
        regime_series.append(state["regime_noise_level"])

    # run gate suite
    report = run_gate_suite(
        z_price=np.array(z_price_series),
        z_force=np.array(z_force_series),
        regime_series=np.array(regime_series),
        returns=returns,
    )

    return {
        "z_price": np.array(z_price_series),
        "z_force": np.array(z_force_series),
        "regime": np.array(regime_series),
        "gate_report": report,
    }


def prepare_input_df(df):
    """
    df 需至少包含: Open, High, Low, Close, Vol
    """
    df = df.copy()

    # === Force proxy（範例，用你現有的邏輯即可）===
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    hl_ret = (df["Close"] - df["Open"]) / (rng + 1e-9)
    pseudo_delta = hl_ret * df["Vol"]

    # ⚠️ 建議：online 標準化（這裡用簡化 rolling）
    m = pseudo_delta.rolling(200).mean()
    s = pseudo_delta.rolling(200).std().replace(0, 1)
    df["force_proxy"] = ((pseudo_delta - m) / s).fillna(0.0)
    return df


def build_params():
    params = {
        "force": {
            "Q_force_base": np.diag([3e-5, 1e-5, 1e-5]),
            "R_force_base": 1e-1,
            "force_proxy_phi": 0.97,
            "q_scale": 0.3,
            "r_scale": 0.3,
            "force_trend_rho": 0.97,
        },
        "price": {
            "Q_price_base": np.diag([1e-6, 1e-7, 1e-8]),
            "R_price_base": (0.001) ** 2,
            "q_scale": 0.3,
            "r_scale": 0.3,
            "q_force_scale": 0.15,
            "force_scale_clip": 5.0,
            "accel_inject": 0.0,  # 默认关：避免 bias
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


def main():
    # === 1) 讀資料 ===
    df = pd.read_csv(
        "/Users/zen/code/bayes_excersice/reports/1216_071632_pepeusdt_1min_fun1000.0cap100.0atr60bw20up60lw60hmm3_cut0.975pnl3.0ext1.0stp5.csv"
    )
    # df = build_synthetic_market_ohlcv(T=2000)
    df = prepare_input_df(df)

    # === 2) 參數 ===
    params, regime_params = build_params()

    # === 3) 跑 MOSAIC Runner ===
    result = run_mosaic_runner(
        df=df,
        mosaic_init_fn=init_mosaic_state,
        mosaic_step_fn=mosaic_step,
        params=params,
        regime_params=regime_params,
    )

    # === 4) 看 Gate Verdict ===
    gate = result["gate_report"]

    print("\n========== MOSAIC Gate Report ==========")
    for k, v in gate.items():
        print(f"{k:30s}: {v}")

    print("\n=== Hard Gates ===")
    print("Gate A (extremes):", "PASS" if gate["gate_A_pass"] else "FAIL")
    print("Gate B (whiteness):", "PASS" if gate["gate_B_pass"] else "FAIL")

    if not gate["gate_A_pass"] or not gate["gate_B_pass"]:
        print("\n❌ MOSAIC FAILED — do NOT proceed to BOCPD / HMM")
    else:
        print("\n✅ MOSAIC PASSED — state space is admissible")


def append_mosaic_state_if_empty(
    df,
    idx,
    mosaic_state,
    mosaic_diag,
):
    """
    Append MOSAIC outputs into df at index `idx`,
    ONLY if the target cells are empty (NaN).

    This function is SAFE for keep-monitoring loops
    and can be called repeatedly.

    Required columns (auto-created if missing):
      pc, pt_speed, pt_accel,
      force_imbalance,
      z_price, z_force
    """

    # --- columns we manage ---
    value_map = {
        "pc": mosaic_state["pc"],
        "pt_speed": mosaic_state["pt_speed"],
        "pt_accel": mosaic_state["pt_accel"],
        "force_imbalance": mosaic_state["force_imbalance"],
        "z_price": mosaic_diag["z_price"],
        "z_force": mosaic_diag["z_force"],
    }

    # --- ensure columns exist ---
    for col in value_map.keys():
        if col not in df.columns:
            df[col] = np.nan

    # --- fill ONLY if empty ---
    for col, value in value_map.items():
        if pd.isna(df.at[idx, col]):
            df.at[idx, col] = value

    return df


import numpy as np
import pandas as pd

from mosaic import mosaic_step  # 你现有的实现


def main2():
    # === 1) 讀資料 ===
    path = "/Users/zen/code/bayes_excersice/reports/1216_085945_pepeusdt_1min_fun1000.0cap100.0atr60bw20up60lw60hmm3_cut0.975pnl3.0ext1.0stp5.csv"
    df = pd.read_csv(path)

    # === 2) 確保 MOSAIC 欄位存在（不覆寫） ===
    mosaic_cols = [
        "m_pc",
        "m_pt_speed",
        "m_pt_accel",
        "m_force",
        "m_force_trend",
        "m_force_bias",
        "m_z_price",
        "m_z_force",
        "pseudo_delta",
    ]
    for c in mosaic_cols:
        if c not in df.columns:
            df[c] = np.nan

    # === 3) 初始化 MOSAIC State / Cov ===
    # 找第一筆還沒算過的 row
    df = prepare_input_df(df)
    start_idx = df.index[df["m_pc"].isna()][0]

    MOSAIC_State = {
        "pc": float(df.loc[start_idx, "Close"]),
        "pt_speed": 0.0,
        "pt_accel": 0.0,
        "force_imbalance": 0.0,
        "force_imbalance_trend": 0.0,
        "force_proxy_bias": 0.0,
        "regime_noise_level": 1.0,
    }
    MOSAIC_Cov = {
        "P_price": np.eye(3) * 10,
        "P_force": np.eye(3) * 5,
    }

    # === 4) MOSAIC 參數（用你已驗證那組） ===
    params = {
        "price": {
            "Q_price_base": np.diag([1e-3, 1e-4, 1e-5]),
            "R_price_base": 1e-1,
            "q_scale": 0.3,
            "r_scale": 0.3,
            "q_force_scale": 0.15,
            "force_scale_clip": 5.0,
            "accel_inject": 0.0,  # 保持 honest
        },
        "force": {
            "Q_force_base": np.diag([3e-5, 1e-5, 1e-5]),
            "R_force_base": 1e-1,
            "q_scale": 0.3,
            "r_scale": 0.3,
            "force_trend_rho": 0.97,
        },
    }
    # === 5) 主迴圈：只補 NaN 的 row ===
    for i in range(start_idx, len(df)):
        if not pd.isna(df.loc[i, "m_pc"]):
            continue  # 已算過就跳過

        obs_close = float(df.loc[i, "Close"])

        # 你原本用的 force proxy（例：pseudo_delta）
        obs_force = float(df.loc[i, "force_proxy"])
        MOSAIC_State, MOSAIC_Cov, diag = mosaic_step(
            MOSAIC_State,
            MOSAIC_Cov,
            obs_close=obs_close,
            obs_force_proxy=obs_force,
            params=params,
        )
        if diag["safe_skip"]:
            continue
        # === 6) 寫回 df（只寫 MOSAIC 欄位） ===
        df.loc[i, "m_pc"] = MOSAIC_State["pc"]
        df.loc[i, "m_pt_speed"] = MOSAIC_State["pt_speed"]
        df.loc[i, "m_pt_accel"] = MOSAIC_State["pt_accel"]
        df.loc[i, "m_force"] = MOSAIC_State["force_imbalance"]
        df.loc[i, "m_force_trend"] = MOSAIC_State["force_imbalance_trend"]
        df.loc[i, "m_force_bias"] = MOSAIC_State["force_proxy_bias"]
        df.loc[i, "m_z_price"] = diag["z_price"]
        df.loc[i, "m_z_force"] = diag["z_force"]

    # === 7) 存回（或回傳給上游） ===
    # df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = main2()
    df.to_csv("out.csv", index=False)
