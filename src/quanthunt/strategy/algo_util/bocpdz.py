from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import gammaln


def logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(a - m))))


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass(frozen=True)
class BOCPDOutputs:
    cp_prob: float
    run_length_mean: float
    run_length_mode: int
    support: int

    log_pred_mix: float
    surprise: float
    z_mix: float
    z2_ewma: float
    surprise_ewma: float

    role: str

    scale_mix: float
    tail_prob_k: float
    shock_score: float

    regime_confidence: float
    risk_level: float


class BOCPDBase:
    """
    Deterministic BOCPD core:
    - Keeps true runlength values in self.r_vals (IMPORTANT)
    - Prunes deterministically by top-K probability (stable tie-break)
    - Always keeps r=0 bucket
    - Diagnostics use prior-predictive for r=0 (not hardcoded zeros)
    """

    def __init__(
        self,
        hazard: float = 0.01,
        r_max: int = 300,
        prune_threshold: float = 1e-8,  # kept for compatibility; not used as hard mask gate
        ewma_alpha: float = 0.05,
        z_clip: float = 50.0,
        debug: bool = False,
    ):
        self.hazard = float(hazard)
        self.r_max = int(r_max)
        self.prune_threshold = float(prune_threshold)
        self.ewma_alpha = float(ewma_alpha)
        self.z_clip = float(z_clip)
        self.debug = bool(debug)

        # Posterior over hypotheses (log-space)
        self.log_R = np.array([0.0], dtype=float)

        # True runlength values aligned with log_R
        self.r_vals = np.array([0], dtype=np.int64)

        self._t = 0
        self._surprise_ewma = 0.0
        self._z2_ewma = 0.0

        # aux aligned with *current* hypotheses (same length as log_R)
        self._last_log_pred = np.array([0.0], dtype=float)
        self._last_z = np.array([0.0], dtype=float)
        self._last_scale = np.array([1.0], dtype=float)

    # ---- Public API ----
    def update(self, x: float) -> BOCPDOutputs:
        x = float(x)
        log_pred, z, scale = self._log_likelihood_and_aux(x)  # over CURRENT hypotheses

        self._update_runlength(x, log_pred, z, scale)
        self._expand_stats(x)
        self._normalize_prune_truncate()

        out = self._outputs(role=self._role_name())
        self._t += 1
        return out

    # ---- Core recursion ----
    def _update_runlength(
        self,
        x: float,
        log_pred: np.ndarray,
        z: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        log_pred = np.asarray(log_pred, dtype=float)

        # new hypotheses count = old + 1 (r=0 plus growth)
        log_R_new = np.full(len(self.log_R) + 1, -np.inf, dtype=float)

        log_h = np.log(self.hazard)
        log_1mh = np.log(1.0 - self.hazard)

        # growth: r -> r+1
        log_R_new[1:] = self.log_R + log_pred + log_1mh

        # changepoint: aggregate to r=0
        log_R_new[0] = logsumexp(self.log_R + log_pred + log_h)

        self.log_R = log_R_new

        # true runlength values
        r_new = np.empty(len(self.r_vals) + 1, dtype=np.int64)
        r_new[0] = 0
        r_new[1:] = self.r_vals + 1
        self.r_vals = r_new

        # aux aligned with new hypotheses:
        # index 0 is fresh segment prior-predictive at x
        lp0, z0, s0 = self._fresh_aux_at_x(x)
        self._last_log_pred = np.concatenate(([lp0], log_pred))
        self._last_z = np.concatenate(([z0], z))
        self._last_scale = np.concatenate(([s0], scale))

    def _normalize_prune_truncate(self) -> None:
        # normalize
        self.log_R = self.log_R - logsumexp(self.log_R)
        R = np.exp(self.log_R)

        # --- deterministic prune/truncate by top-K probability ---
        # Always keep r=0 bucket (index where r_vals == 0, should be 0)
        idx0 = int(np.where(self.r_vals == 0)[0][0])

        # stable sorting by (-R, r_vals, index) to remove tie nondeterminism
        idx = np.arange(len(R), dtype=np.int64)
        order = np.lexsort((idx, self.r_vals, -R))  # primary: -R, tie: r_vals, tie: idx

        # choose top K (bounded by r_max)
        k = min(self.r_max, len(R))
        keep = order[:k]

        # ensure idx0 kept
        if idx0 not in keep:
            # replace last with idx0 deterministically
            keep = np.array(list(keep[:-1]) + [idx0], dtype=np.int64)

        # Now: for nicer semantics (and stable prints), sort keep by r_vals ascending
        keep = keep[np.argsort(self.r_vals[keep], kind="mergesort")]

        if len(keep) < len(R):
            self._apply_keep_indices(keep)
            R = np.exp(self.log_R)

        # renormalize again (exact)
        self.log_R = self.log_R - logsumexp(self.log_R)

    def _apply_keep_indices(self, keep: np.ndarray) -> None:
        keep = np.asarray(keep, dtype=np.int64)

        self.log_R = self.log_R[keep]
        self.r_vals = self.r_vals[keep]

        self._last_log_pred = self._last_log_pred[keep]
        self._last_z = self._last_z[keep]
        self._last_scale = self._last_scale[keep]

        # stats: keep by mask to preserve your existing signature
        mask = np.zeros(len(keep) + (0), dtype=bool)  # placeholder (not used)
        # We must build mask over original length; easiest: pass keep to new hook.
        self._keep_stats(keep)

    # ---- Outputs ----
    def _outputs(self, role: str) -> BOCPDOutputs:
        R = np.exp(self.log_R)

        cp_prob = float(R[self._idx_r0()])
        support = int(len(R))

        # runlength mean/mode computed on TRUE runlength values
        run_length_mean = float(np.sum(self.r_vals.astype(float) * R))
        mode_idx = int(np.argmax(R))
        run_length_mode = int(self.r_vals[mode_idx])

        log_pred_mix = float(np.sum(R * self._last_log_pred))
        surprise = float(-log_pred_mix)

        z_mix = float(np.sum(R * self._last_z))
        z_mix = float(np.clip(z_mix, -self.z_clip, self.z_clip))

        # ADD: robust magnitude (no cancel)
        z_rms = float(np.sqrt(np.sum(R * (self._last_z**2))))
        z_rms = float(np.clip(z_rms, 0.0, self.z_clip))

        scale_mix = float(np.sum(R * self._last_scale))

        a = self.ewma_alpha
        self._surprise_ewma = (1 - a) * self._surprise_ewma + a * surprise
        self._z2_ewma = (1 - a) * self._z2_ewma + a * (z_mix * z_mix)

        # concentration proxy (entropy over hypotheses)
        entropy = -float(np.sum(R * np.log(R + 1e-12)))
        entropy_norm = entropy / np.log(max(support, 2))
        confidence = clip01((1.0 - entropy_norm) * np.exp(-0.15 * self._surprise_ewma))

        tail_prob_k, shock_score, risk_level = self._risk_semantics(R, z_rms, scale_mix)

        if self.debug and (cp_prob > 0.05 or self._t % 200 == 0):
            print(
                f"[DBG {role}] t={self._t:4d} "
                f"cp={cp_prob:.4f} mode_r={run_length_mode:4d} mean_r={run_length_mean:7.2f} "
                f"surp={surprise:.3f} surpEWMA={self._surprise_ewma:.3f} z={z_mix:.2f} "
                f"support={support}"
            )

        return BOCPDOutputs(
            cp_prob=cp_prob,
            run_length_mean=run_length_mean,
            run_length_mode=run_length_mode,
            support=support,
            log_pred_mix=log_pred_mix,
            surprise=surprise,
            z_mix=z_mix,
            z2_ewma=float(self._z2_ewma),
            surprise_ewma=float(self._surprise_ewma),
            role=role,
            scale_mix=scale_mix,
            tail_prob_k=tail_prob_k,
            shock_score=shock_score,
            regime_confidence=confidence,
            risk_level=risk_level,
        )

    def _idx_r0(self) -> int:
        # r=0 should exist and be unique
        return int(np.where(self.r_vals == 0)[0][0])

    # ---- Subclass hooks ----
    def _role_name(self) -> str:
        raise NotImplementedError

    def _log_likelihood_and_aux(
        self, x: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _expand_stats(self, x: float) -> None:
        raise NotImplementedError

    # NEW: keep stats by indices (deterministic prune uses indices, not mask)
    def _keep_stats(self, keep: np.ndarray) -> None:
        raise NotImplementedError

    # prior predictive aux for fresh segment at x (diagnostics only)
    def _fresh_aux_at_x(self, x: float) -> tuple[float, float, float]:
        raise NotImplementedError

    def _risk_semantics(
        self, R: np.ndarray, z_mix: float, scale_mix: float
    ) -> tuple[float, float, float]:
        return (float("nan"), float("nan"), 0.0)


class BOCPDGaussianG0(BOCPDBase):
    """
    Gaussian G0:
    x ~ Normal(m, sigma_obs^2 + v)
    Posterior over mean: m, v (known sigma_obs^2)
    """

    def __init__(
        self,
        sigma_obs: float = 1.0,
        mu0: float = 0.0,
        var0: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma_obs2 = float(sigma_obs) ** 2
        self.mu0 = float(mu0)
        self.var0 = float(var0)

        self.m = np.array([self.mu0], dtype=float)
        self.v = np.array([self.var0], dtype=float)
        self.counts = np.array([1.0], dtype=float)

        self.gate_on = 2.5  # surprise_ewma 進入門檻（你可調）
        self.gate_off = 1.8  # surprise_ewma 退出門檻（滯後）
        self.hold_ticks = 40  # 進入後至少維持 N 根
        self._gate_hold = 0
        self._gate_state = 0  # 0/1
        self.hazard_mul_min = 0.05
        self.hazard_mul_max = 8.0

    def _role_name(self) -> str:
        return "Gaussian_G0"

    def _log_likelihood_and_aux(self, x: float):
        pred_var = np.maximum(self.sigma_obs2 + self.v, 1e-12)
        pred_sigma = np.sqrt(pred_var)

        log_pred = -0.5 * (
            np.log(2 * np.pi * pred_var) + ((x - self.m) ** 2) / pred_var
        )

        z = (x - self.m) / (pred_sigma + 1e-12)
        z = np.clip(z, -self.z_clip, self.z_clip)

        return log_pred, z, pred_sigma

    def _expand_stats(self, x: float) -> None:
        new_len = len(self.log_R)

        new_m = np.empty(new_len, dtype=float)
        new_v = np.empty(new_len, dtype=float)
        new_counts = np.empty(new_len, dtype=float)

        # fresh segment: prior -> posterior with one obs
        v0 = self.var0
        m0 = self.mu0
        v_post = 1.0 / (1.0 / v0 + 1.0 / self.sigma_obs2)
        m_post = v_post * (m0 / v0 + x / self.sigma_obs2)

        new_m[0] = m_post
        new_v[0] = v_post
        new_counts[0] = 1.0

        for i in range(len(self.m)):
            m, v = self.m[i], self.v[i]
            v_post = 1.0 / (1.0 / v + 1.0 / self.sigma_obs2)
            m_post = v_post * (m / v + x / self.sigma_obs2)

            new_m[i + 1] = m_post
            new_v[i + 1] = v_post
            new_counts[i + 1] = self.counts[i] + 1.0

        self.m, self.v, self.counts = new_m, new_v, new_counts

    def _keep_stats(self, keep: np.ndarray) -> None:
        self.m = self.m[keep]
        self.v = self.v[keep]
        self.counts = self.counts[keep]

    def _fresh_aux_at_x(self, x: float) -> tuple[float, float, float]:
        # prior predictive: x ~ N(mu0, sigma_obs2 + var0)
        pred_var = max(self.sigma_obs2 + self.var0, 1e-12)
        pred_sigma = float(np.sqrt(pred_var))
        lp = float(
            -0.5 * (np.log(2 * np.pi * pred_var) + ((x - self.mu0) ** 2) / pred_var)
        )
        z = float((x - self.mu0) / (pred_sigma + 1e-12))
        z = float(np.clip(z, -self.z_clip, self.z_clip))
        return lp, z, pred_sigma


class BOCPDStudentTP1(BOCPDBase):
    """
    Student-t P1 with NIG prior.
    Predictive is Student-t with df=2*alpha, loc=mu, scale=sqrt(beta*(kappa+1)/(alpha*kappa))
    """

    def __init__(
        self,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 2.0,
        beta0: float = 2.0,
        k_tail: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.k_tail = float(k_tail)

        self.mu = np.array([self.mu0], dtype=float)
        self.kappa = np.array([self.kappa0], dtype=float)
        self.alpha = np.array([self.alpha0], dtype=float)
        self.beta = np.array([self.beta0], dtype=float)
        self.counts = np.array([1.0], dtype=float)

    def _role_name(self) -> str:
        return "StudentT_P1"

    def _predictive_params(self):
        df = 2.0 * self.alpha
        scale2 = (self.beta * (self.kappa + 1.0)) / (self.alpha * self.kappa + 1e-12)
        scale2 = np.maximum(scale2, 1e-12)
        return df, self.mu, np.sqrt(scale2)

    def _log_likelihood_and_aux(self, x: float):
        df, loc, scale = self._predictive_params()
        z = (x - loc) / (scale + 1e-12)
        z = np.clip(z, -self.z_clip, self.z_clip)

        log_norm = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * (np.log(df) + np.log(np.pi))
        )
        log_kernel = -((df + 1.0) / 2.0) * np.log(1.0 + (z * z) / df)
        log_pred = log_norm + log_kernel - np.log(scale + 1e-12)

        return log_pred, z, scale

    def _expand_stats(self, x: float) -> None:
        new_len = len(self.log_R)

        new_mu = np.empty(new_len, dtype=float)
        new_kappa = np.empty(new_len, dtype=float)
        new_alpha = np.empty(new_len, dtype=float)
        new_beta = np.empty(new_len, dtype=float)
        new_counts = np.empty(new_len, dtype=float)

        # fresh
        mu, kappa, alpha, beta = self.mu0, self.kappa0, self.alpha0, self.beta0
        kappa2 = kappa + 1.0
        mu2 = (kappa * mu + x) / kappa2
        alpha2 = alpha + 0.5
        beta2 = beta + 0.5 * (kappa * (x - mu) ** 2) / kappa2
        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = mu2, kappa2, alpha2, beta2
        new_counts[0] = 1.0

        for i in range(len(self.mu)):
            mu, kappa, alpha, beta = (
                self.mu[i],
                self.kappa[i],
                self.alpha[i],
                self.beta[i],
            )
            kappa2 = kappa + 1.0
            mu2 = (kappa * mu + x) / kappa2
            alpha2 = alpha + 0.5
            beta2 = beta + 0.5 * (kappa * (x - mu) ** 2) / kappa2
            new_mu[i + 1], new_kappa[i + 1], new_alpha[i + 1], new_beta[i + 1] = (
                mu2,
                kappa2,
                alpha2,
                beta2,
            )
            new_counts[i + 1] = self.counts[i] + 1.0

        self.mu, self.kappa, self.alpha, self.beta, self.counts = (
            new_mu,
            new_kappa,
            new_alpha,
            new_beta,
            new_counts,
        )

    def _keep_stats(self, keep: np.ndarray) -> None:
        self.mu = self.mu[keep]
        self.kappa = self.kappa[keep]
        self.alpha = self.alpha[keep]
        self.beta = self.beta[keep]
        self.counts = self.counts[keep]

    def _fresh_aux_at_x(self, x: float) -> tuple[float, float, float]:
        # prior predictive at x
        df = 2.0 * self.alpha0
        scale2 = (self.beta0 * (self.kappa0 + 1.0)) / (
            self.alpha0 * self.kappa0 + 1e-12
        )
        scale = float(np.sqrt(max(scale2, 1e-12)))
        z = float((x - self.mu0) / (scale + 1e-12))
        z = float(np.clip(z, -self.z_clip, self.z_clip))

        log_norm = float(
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * (np.log(df) + np.log(np.pi))
        )
        log_kernel = float(-((df + 1.0) / 2.0) * np.log(1.0 + (z * z) / df))
        lp = float(log_norm + log_kernel - np.log(scale + 1e-12))
        return lp, z, scale

    def _risk_semantics(
        self, R: np.ndarray, z_mix: float, scale_mix: float
    ) -> tuple[float, float, float]:
        shock = 1.0 - np.exp(-0.35 * abs(z_mix))
        k = self.k_tail
        tail = 1.0 / (1.0 + np.exp(-(abs(z_mix) - k)))
        scale_term = 1.0 - np.exp(-0.25 * max(scale_mix, 1e-9))
        risk = clip01(0.55 * tail + 0.35 * shock + 0.10 * scale_term)
        return float(tail), float(shock), float(risk)


class DualBOCPD:
    """
    Runs:
    - Gaussian G0 for regime death evidence
    - Student-t P1 for risk evidence
    Returns both outputs each tick.
    """

    def __init__(self, g0: BOCPDGaussianG0, p1: BOCPDStudentTP1):
        self.g0 = g0
        self.p1 = p1

    def update(self, x: float) -> tuple[BOCPDOutputs, BOCPDOutputs]:
        out_g0 = self.g0.update(x)
        out_p1 = self.p1.update(x)
        return out_g0, out_p1


class DualBOCPDWrapper:
    def __init__(self, dual_bocpd, phase_fsm: BOCPDPhaseFSM, x_col="m_z_mix"):
        self.dual = dual_bocpd
        self.fsm = phase_fsm
        self.x_col = x_col

        self._last_out_g0 = None
        self._last_out_p1 = None

    def update_df_row(self, df, idx):
        x = float(df.at[idx, self.x_col])

        # If Phase4 freeze, do NOT update engines anymore
        if (
            self.fsm.phase == 4
            and self.fsm.cfg.on_phase4 == "freeze"
            and self._last_out_g0 is not None
        ):
            out_g0, out_p1 = self._last_out_g0, self._last_out_p1
            phase = 4
        else:
            out_g0, out_p1 = self.dual.update(x)
            phase = self.fsm.update(out_g0, out_p1, hazard_g0=self.dual.g0.hazard)
            self._last_out_g0, self._last_out_p1 = out_g0, out_p1

        # write snapshot
        df.at[idx, "bocpd_phase"] = phase

        df.at[idx, "bocpd_cp_prob"] = out_g0.cp_prob
        df.at[idx, "bocpd_runlen_mean"] = out_g0.run_length_mean
        df.at[idx, "bocpd_runlen_mode"] = out_g0.run_length_mode

        df.at[idx, "bocpd_risk"] = out_p1.risk_level
        df.at[idx, "bocpd_tail"] = out_p1.tail_prob_k
        df.at[idx, "bocpd_shock"] = out_p1.shock_score

        df.at[idx, "bocpd_surpEWMA"] = out_g0.surprise_ewma
        df.at[idx, "bocpd_cp_excess"] = float(out_g0.cp_prob) / max(
            self.dual.g0.hazard, 1e-12
        )

        df.at[idx, "bocpd_model"] = "G0+P1"

    def run_online(self, df):
        """
        Convenience: run from first to last row once.
        Assumes df grows over time; safe for backfill.
        """
        for idx in df.index:
            self.update_df_row(df, idx)


@dataclass
class PhaseFSMConfig:
    # --- warmup ---
    warmup_ticks: int = 60  # 1D 建議 30~90；1m 建議 200~800

    # --- P1 risk thresholds (early warning) ---
    r1_watch: float = 0.60  # phase0->1
    r2_caution: float = 0.75  # phase1/0->2
    sustain_p1: int = 5  # 連續幾根才升級到 phase2
    cooldown_p1: int = 10  # 風險下降後，phase1/2 不會降，但可清計數用

    # --- G0 cp thresholds (regime death evidence) ---
    cp_excess_pre: float = 2.5  # cp_prob / hazard > 2.5 -> phase3 (Pre-CP)
    cp_excess_confirm: float = 5.0  # cp_prob / hazard > 5.0 sustained -> phase4
    sustain_g0: int = 3

    # --- optional: confirm needs additional stress evidence ---
    require_surprise: bool = True
    surprise_ewma_pre: float = 1.2
    surprise_ewma_confirm: float = 1.6

    # --- Phase4 behavior ---
    on_phase4: str = "freeze"  # "freeze" | "reset"
    reset_cooldown: int = (
        30  # if on_phase4=="reset": wait N ticks before allowing phase>0 again
    )


class BOCPDPhaseFSM:
    """
    Phase:
      0 Stable
      1 Watch   (P1 early warning)
      2 Caution (P1 sustained high risk)
      3 Pre-CP  (G0 evidence rising vs hazard baseline)
      4 CP Confirmed (G0 sustained strong evidence) - absorbing unless reset mode

    Key fixes:
    - warmup gate: no phase escalation during early ticks
    - cp_prob is judged relative to hazard baseline (cp_excess)
    - phase4 has real policy effect (freeze or reset)
    """

    def __init__(self, cfg: PhaseFSMConfig):
        self.cfg = cfg
        self.phase = 0

        self._t = 0
        self._p1_hi_cnt = 0
        self._g0_hi_cnt = 0

        # reset mode state
        self._in_reset_cooldown = 0

    def _cp_excess(self, cp_prob: float, hazard: float) -> float:
        hazard = max(float(hazard), 1e-12)
        return float(cp_prob) / hazard

    def update(self, out_g0, out_p1, hazard_g0: float) -> int:
        """
        out_g0/out_p1 are BOCPDOutputs from your system.
        hazard_g0 should be the constant hazard used in G0 instance.
        """

        self._t += 1

        # ---- reset cooldown mode ----
        if self._in_reset_cooldown > 0:
            self._in_reset_cooldown -= 1
            self.phase = 0
            return self.phase

        # ---- absorbing behavior ----
        if self.phase == 4 and self.cfg.on_phase4 == "freeze":
            return 4

        # ---- warmup gate ----
        if self._t <= self.cfg.warmup_ticks:
            self.phase = 0
            # still update counters lightly (optional), but do not escalate
            self._p1_hi_cnt = 0
            self._g0_hi_cnt = 0
            return self.phase

        # ---- P1 drives early warning phases ----
        # phase0->1: immediate watch if risk crosses r1
        if self.phase < 1 and out_p1.risk_level > self.cfg.r1_watch:
            self.phase = 1

        # sustain for phase2
        if out_p1.risk_level > self.cfg.r2_caution:
            self._p1_hi_cnt += 1
        else:
            # risk fell; don't downgrade phase, just clear counter slowly
            self._p1_hi_cnt = max(0, self._p1_hi_cnt - 1)

        if self.phase < 2 and self._p1_hi_cnt >= self.cfg.sustain_p1:
            self.phase = 2

        # ---- G0 drives Pre-CP / Confirmed CP ----
        cp_excess = self._cp_excess(out_g0.cp_prob, hazard_g0)

        # Optional: require "stress" evidence besides cp_excess
        stress_pre = True
        stress_confirm = True
        if self.cfg.require_surprise:
            stress_pre = out_g0.surprise_ewma >= self.cfg.surprise_ewma_pre
            stress_confirm = out_g0.surprise_ewma >= self.cfg.surprise_ewma_confirm

        # phase3 (Pre-CP): cp_excess crosses pre threshold (and stress)
        if self.phase < 3 and (cp_excess >= self.cfg.cp_excess_pre) and stress_pre:
            self.phase = 3

        # phase4 (Confirmed): sustained high cp_excess (and stress)
        if (cp_excess >= self.cfg.cp_excess_confirm) and stress_confirm:
            self._g0_hi_cnt += 1
        else:
            self._g0_hi_cnt = max(0, self._g0_hi_cnt - 1)

        if self._g0_hi_cnt >= self.cfg.sustain_g0:
            self.phase = 4
            if self.cfg.on_phase4 == "reset":
                # Enter cooldown; caller can also reset BOCPD engines if desired
                self._in_reset_cooldown = self.cfg.reset_cooldown
                self.phase = 0  # immediately back to 0 after reset trigger

        return self.phase
