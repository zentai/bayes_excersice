import numpy as np

class BOCPD:
    """
    Bayesian Online Change Point Detection (Minimal Prototype)

    - Input: z_mix (whitened, stationary scalar)
    - Output: cp_prob, run-length stats, hard_cp_flag
    - Likelihood: Gaussian (can be replaced by Student-t)
    - Hazard: constant or noise-modulated
    """

    def __init__(
        self,
        hazard=0.01,
        r_max=200,
        prune_threshold=1e-6,
        cp_threshold=0.5,
        cp_confirm_steps=3,
        init_mean=0.0,
        init_var=1.0,
    ):
        # --- BOCPD core state ---
        self.r_max = r_max
        self.hazard_base = hazard
        self.prune_threshold = prune_threshold

        # run-length posterior (log-space)
        self.log_R = np.array([0.0])  # log P(r=0) = 1 initially

        # sufficient statistics per run-length
        # here: simple Gaussian with online mean/var
        self.means = np.array([init_mean])
        self.vars = np.array([init_var])
        self.counts = np.array([1])

        # --- regime bookkeeping ---
        self.regime_id = 0
        self.cp_threshold = cp_threshold
        self.cp_confirm_steps = cp_confirm_steps
        self.cp_counter = 0

    # =====================================================
    # public API
    # =====================================================
    def update(self, x_t, aux=None):
        """
        One-step BOCPD update.
        """
        hazard_t = self._compute_hazard(aux)
        log_pred = self._predictive_loglik(x_t)

        self._update_run_length_posterior(log_pred, hazard_t)
        self._normalize_and_prune()
        self._update_sufficient_statistics(x_t)

        outputs = self._compute_outputs()
        self._update_regime_id(outputs["cp_prob"])

        return outputs

    # =====================================================
    # sub-functions
    # =====================================================
    def _compute_hazard(self, aux):
        """
        Hazard function.
        Default: constant.
        """
        if aux is None:
            return self.hazard_base

        noise = aux.get("regime_noise", None)
        if noise is None:
            return self.hazard_base

        # simple modulation (safe default)
        return np.clip(self.hazard_base * (1.0 + noise), 1e-4, 0.5)

    def _predictive_loglik(self, x_t):
        """
        Predictive log-likelihood for each run-length.
        Gaussian version (replaceable).
        """
        var = self.vars + 1e-8
        return -0.5 * (
            np.log(2 * np.pi * var)
            + (x_t - self.means) ** 2 / var
        )

    def _update_run_length_posterior(self, log_pred, hazard):
        """
        Core BOCPD recursion (log-space).
        """
        log_R_new = np.full(len(self.log_R) + 1, -np.inf)

        # growth probabilities: r -> r+1
        log_R_new[1:] = (
            self.log_R
            + log_pred
            + np.log(1.0 - hazard)
        )

        # change point probability: r -> 0
        log_R_new[0] = np.logaddexp.reduce(
            self.log_R + log_pred + np.log(hazard)
        )

        self.log_R = log_R_new

    def _normalize_and_prune(self):
        """
        Normalize posterior and prune low-mass run-lengths.
        """
        # normalize
        log_norm = np.logaddexp.reduce(self.log_R)
        self.log_R -= log_norm

        R = np.exp(self.log_R)

        # prune
        mask = R > self.prune_threshold
        self.log_R = self.log_R[mask]
        self.means = self.means[mask]
        self.vars = self.vars[mask]
        self.counts = self.counts[mask]

        # cap run-length
        if len(self.log_R) > self.r_max:
            self.log_R = self.log_R[:self.r_max]
            self.means = self.means[:self.r_max]
            self.vars = self.vars[:self.r_max]
            self.counts = self.counts[:self.r_max]

    def _update_sufficient_statistics(self, x_t):
        """
        Online update of mean / variance for each run-length.
        """
        # prepend new CP stats
        new_means = np.empty(len(self.means) + 1)
        new_vars = np.empty(len(self.vars) + 1)
        new_counts = np.empty(len(self.counts) + 1)

        # r = 0 (new regime)
        new_means[0] = x_t
        new_vars[0] = 1.0
        new_counts[0] = 1

        # r > 0
        for i in range(len(self.means)):
            n = self.counts[i]
            mean = self.means[i]
            var = self.vars[i]

            n_new = n + 1
            mean_new = mean + (x_t - mean) / n_new
            var_new = (
                (n - 1) * var + (x_t - mean) * (x_t - mean_new)
            ) / max(n, 1)

            new_means[i + 1] = mean_new
            new_vars[i + 1] = max(var_new, 1e-6)
            new_counts[i + 1] = n_new

        self.means = new_means
        self.vars = new_vars
        self.counts = new_counts

    def _compute_outputs(self):
        """
        Translate posterior into system-friendly outputs.
        """
        R = np.exp(self.log_R)
        cp_prob = R[0]

        run_lengths = np.arange(len(R))
        run_length_mean = np.sum(run_lengths * R)
        run_length_mode = int(run_lengths[np.argmax(R)])

        return {
            "cp_prob": float(cp_prob),
            "run_length_mean": float(run_length_mean),
            "run_length_mode": run_length_mode,
            "hard_cp_flag": False,  # updated later
            "regime_id": self.regime_id,
        }

    def _update_regime_id(self, cp_prob):
        """
        Anti-flap CP confirmation.
        """
        if cp_prob > self.cp_threshold:
            self.cp_counter += 1
        else:
            self.cp_counter = 0

        if self.cp_counter >= self.cp_confirm_steps:
            self.regime_id += 1
            self.cp_counter = 0
