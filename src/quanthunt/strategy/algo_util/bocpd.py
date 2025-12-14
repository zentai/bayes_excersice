import numpy as np
from scipy.special import gammaln
from typing import Dict


def student_t_logpdf(
    x: np.ndarray,
    mu: np.ndarray,
    scale: np.ndarray,
    nu: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    數值穩定的 Student-t log pdf，支援 broadcasting。

    x, mu, scale, nu 皆可是可廣播的 array。
    回傳：log p(x | mu, scale, nu)
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    scale = np.asarray(scale, dtype=float)
    nu = np.asarray(nu, dtype=float)

    scale = np.maximum(scale, eps)
    nu = np.maximum(nu, eps)

    z = (x - mu) / scale

    log_coeff = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * (np.log(nu * np.pi + eps) + 2.0 * np.log(scale))
    )

    log_pdf = log_coeff - (nu + 1.0) / 2.0 * np.log1p((z**2) / nu)

    # 防止極端值炸掉後續 logsumexp
    log_pdf = np.clip(log_pdf, -1e6, 1e6)
    return log_pdf


def logsumexp(a: np.ndarray) -> float:
    """
    數值穩定版 log(sum(exp(a)))。
    """
    a = np.asarray(a, dtype=float)
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return -np.inf
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def bocpd_student_t_trunc(
    x: np.ndarray,
    hazard_lambda: float = 200.0,
    max_run_length: int = 300,
    mu0: float = 0.0,
    kappa0: float = 1e-2,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Bayesian Online Change Point Detection (BOCPD)
    使用 Normal-Inverse-Gamma → Student-t 的共軛先驗，
    並採用 run-length truncation + log-domain 計算。

    參數
    ----
    x : 1D np.ndarray
        時間序列（通常是 log-return）。
    hazard_lambda : float
        常數 hazard 的 lambda，代表平均 regime 長度。
        baseline P(change) ~= 1 / hazard_lambda。
    max_run_length : int
        最大 run-length（超過就 truncate），避免 R 爆成超長三角矩陣。
    mu0, kappa0, alpha0, beta0 : float
        Normal-Inverse-Gamma 超參數。

    回傳
    ----
    dict with:
        - "log_R"     : shape (T+1, max_R)，log run-length posterior
        - "cp_prob"   : shape (T,)，每一步的 P(r_t = 0 | x_1:t)
        - "run_length": shape (T,)，每一步最可能的 run-length (argmax)
    """
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    if T == 0:
        return {
            "log_R": np.zeros((0, 0)),
            "cp_prob": np.zeros(0),
            "run_length": np.zeros(0, dtype=int),
        }

    max_R = min(max_run_length, T)

    # log_R[t, r] = log P(r_t = r | x_1:t)，多一列 t=0 當起始（還沒看到資料）
    log_R = np.full((T + 1, max_R), -np.inf, dtype=float)
    log_R[0, 0] = 0.0  # t=0, r=0 的先驗機率 = 1

    # Normal-Inverse-Gamma 參數：對每個 run-length r 維護一組
    mu = np.full(max_R, mu0, dtype=float)
    kappa = np.full(max_R, max(kappa0, eps), dtype=float)
    alpha = np.full(max_R, max(alpha0, eps), dtype=float)
    beta = np.full(max_R, max(beta0, eps), dtype=float)

    # 常數 hazard
    # H = 1.0 / float(hazard_lambda)
    # H = min(0.5, 1 / (hazard_lambda + 1))
    # log_H = np.log(H)
    # log_1mH = np.log(1.0 - H)

    cp_prob = np.zeros(T, dtype=float)
    run_length = np.zeros(T, dtype=int)

    for t in range(T):
        x_t = x[t]

        # 目前允許的最大 run-length（考慮 t 與 max_R）
        max_r_t = min(t, max_R - 1)
        idx = np.arange(max_r_t + 1)  # 0..max_r_t

        H_min = 0.002  # 剛啟動時，變盤機率很低
        H_max = 0.10  # 再怎麼久，也不會每根都有 50% 變盤
        lam = float(hazard_lambda)  # 用 hazard_lambda 當時間尺度

        # 平滑遞增 hazard：一開始小，r ~ lam 附近開始抬頭，之後趨近 H_max
        H = H_min + (H_max - H_min) * (1.0 - np.exp(-idx / (lam + 1e-9)))
        H = np.clip(H, H_min, H_max)
        log_H = np.log(H)
        log_1mH = np.log(1.0 - H)

        # ---- Step 1: 用「看到 x_t 前」的參數算 predictive log pdf ----
        a = alpha[idx]
        k = kappa[idx]
        b = beta[idx]

        nu = 2.0 * a
        var = b * (k + 1.0) / (a * k + eps)
        scale = np.sqrt(np.maximum(var, eps))

        log_pred = student_t_logpdf(
            x_t, mu[idx], scale, nu, eps=eps
        )  # shape = (max_r_t+1,)

        # ---- Step 2: 用上一刻的 log_R[t, :] 更新到 log_R[t+1, :] ----
        log_R_prev = log_R[t, : max_r_t + 1]  # shape = (max_r_t+1,)

        # 2a) growth：r_t = r_{t-1} + 1，對 r >= 1
        if max_r_t > 0:
            # r = 1..max_r_t 從 r-1 來
            growth_src = log_R_prev[:-1] + log_1mH[:-1] + log_pred[:-1]
            log_R[t + 1, 1 : max_r_t + 1] = growth_src

        # 2b) change point：r_t = 0，從所有舊的 r 來
        cp_terms = log_R_prev + log_H + log_pred
        log_R[t + 1, 0] = logsumexp(cp_terms)

        # ---- Step 3: normalize log_R[t+1, :] ----
        Z = logsumexp(log_R[t + 1, : max_r_t + 1])
        if not np.isfinite(Z):
            # 防呆：全部掛掉就硬 reset
            log_R[t + 1, :] = -np.inf
            log_R[t + 1, 0] = 0.0
            cp_prob[t] = 1.0
            run_length[t] = 0
        else:
            log_R[t + 1, : max_r_t + 1] -= Z
            cp_prob[t] = np.exp(log_R[t + 1, 0])
            run_length[t] = int(np.argmax(log_R[t + 1, : max_r_t + 1]))

        # ---- Step 4: 更新 NIΓ 參數，準備下一步 ----
        new_mu = np.copy(mu)
        new_kappa = np.copy(kappa)
        new_alpha = np.copy(alpha)
        new_beta = np.copy(beta)

        # r = 0：變盤後的新 regime，reset 回先驗
        new_mu[0] = mu0
        new_kappa[0] = max(kappa0, eps)
        new_alpha[0] = max(alpha0, eps)
        new_beta[0] = max(beta0, eps)

        # r = 1..max_r_t：從 r-1 的 posterior + x_t 更新來
        for r in range(1, max_r_t + 1):
            m_prev = mu[r - 1]
            k_prev = max(kappa[r - 1], eps)
            a_prev = max(alpha[r - 1], eps)
            b_prev = max(beta[r - 1], eps)

            k_new = k_prev + 1.0
            m_new = (k_prev * m_prev + x_t) / k_new
            a_new = a_prev + 0.5
            b_new = b_prev + 0.5 * k_prev * (x_t - m_prev) ** 2 / k_new

            new_mu[r] = m_new
            new_kappa[r] = k_new
            new_alpha[r] = a_new
            new_beta[r] = b_new

        # 超過 max_r_t 的 r 不用動（永遠沒機率，也不會被用到）
        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

    return {
        "log_R": log_R,
        "cp_prob": cp_prob,
        "run_length": run_length,
    }


import pandas as pd


def apply_bocpd_to_df(
    df: pd.DataFrame,
    price_col: str = "Close",
    hazard_lambda: float = 200.0,
    max_run_length: int = 300,
    mu0: float = 0.0,
    kappa0: float = 1e-2,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> pd.DataFrame:
    """
    在 K 線 DataFrame 上套 BOCPD，輸出兩個欄位：
        - cp_prob   : 變盤機率 P(r_t=0 | x_1:t)
        - run_length: 最有可能的 run-length

    你可以後續再自己定義：
        - cp_signal = (cp_prob > 某門檻).astype(int)
    """
    out = df.copy()

    # 1) 用 log-return 當 x（你也可以換成 Kalman / VWAP / slope 等）
    # out["ret"] = np.log(out[price_col] / out[price_col].shift(1))
    out["raw_ret"] = out[price_col] - out[price_col].shift(1)
    roll_std = out["raw_ret"].rolling(window=50, min_periods=20).std()
    out["ret"] = out["raw_ret"] / (roll_std + 1e-9)
    rets = out["ret"].dropna().to_numpy()

    if len(rets) == 0:
        out["cp_prob"] = 0.0
        out["run_length"] = 0
        return out

    res = bocpd_student_t_trunc(
        rets,
        hazard_lambda=hazard_lambda,
        max_run_length=max_run_length,
        mu0=mu0,
        kappa0=kappa0,
        alpha0=alpha0,
        beta0=beta0,
    )

    valid_idx = out.index[out["ret"].notna()]  # 第一筆 ret 從第二根 K 線開始
    out.loc[valid_idx, "cp_prob"] = res["cp_prob"]
    out.loc[valid_idx, "run_length"] = res["run_length"]

    out[["cp_prob", "run_length"]] = out[["cp_prob", "run_length"]].fillna(0)
    df.update(out)
    return df
