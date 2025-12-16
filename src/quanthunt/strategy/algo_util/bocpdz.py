import numpy as np

# =====================================================
# BOCPD class（与你前一版一致，略微整理）
# =====================================================

class BOCPD:
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
        self.r_max = r_max
        self.hazard_base = hazard
        self.prune_threshold = prune_threshold

        self.log_R = np.array([0.0])
        self.means = np.array([init_mean])
        self.vars = np.array([init_var])
        self.counts = np.array([1])

        self.regime_id = 0
        self.cp_threshold = cp_threshold
        self.cp_confirm_steps = cp_confirm_steps
        self.cp_counter = 0

    def update(self, x_t, aux=None):
        hazard_t = self._compute_hazard(aux)
        log_pred = self._predictive_loglik(x_t)

        self._update_run_length_posterior(log_pred, hazard_t)
        self._normalize_and_prune()
        self._update_sufficient_statistics(x_t)

        outputs = self._compute_outputs()
        self._update_regime_id(outputs["cp_prob"])
        outputs["hard_cp_flag"] = (self.cp_counter == 0 and outputs["cp_prob"] > self.cp_threshold)

        return outputs

    def _compute_hazard(self, aux):
        if aux is None:
            return self.hazard_base
        noise = aux.get("regime_noise", None)
        if noise is None:
            return self.hazard_base
        return np.clip(self.hazard_base * (1.0 + noise), 1e-4, 0.5)

    def _predictive_loglik(self, x_t):
        var = self.vars + 1e-8
        return -0.5 * (np.log(2 * np.pi * var) + (x_t - self.means) ** 2 / var)

    def _update_run_length_posterior(self, log_pred, hazard):
        log_R_new = np.full(len(self.log_R) + 1, -np.inf)
        log_R_new[1:] = self.log_R + log_pred + np.log(1.0 - hazard)
        log_R_new[0] = np.logaddexp.reduce(self.log_R + log_pred + np.log(hazard))
        self.log_R = log_R_new

    def _normalize_and_prune(self):
        log_norm = np.logaddexp.reduce(self.log_R)
        self.log_R -= log_norm

        R = np.exp(self.log_R)
        mask = R > self.prune_threshold

        self.log_R = self.log_R[mask]
        self.means = self.means[mask]
        self.vars = self.vars[mask]
        self.counts = self.counts[mask]

        if len(self.log_R) > self.r_max:
            self.log_R = self.log_R[:self.r_max]
            self.means = self.means[:self.r_max]
            self.vars = self.vars[:self.r_max]
            self.counts = self.counts[:self.r_max]
#fixme
def _expand_sufficient_statistics(self, x_t):
    """
    Expand stats to align with new run-lengths.
    Must be called immediately after log_R is updated.
    """
    new_means = np.empty(len(self.log_R))
    new_vars = np.empty(len(self.log_R))
    new_counts = np.empty(len(self.log_R))

    # r = 0 (new regime)
    new_means[0] = x_t
    new_vars[0] = 1.0
    new_counts[0] = 1

    # r > 0 (shift old stats)
    for i in range(len(self.means)):
        n = self.counts[i]
        mean = self.means[i]
        var = self.vars[i]

        n_new = n + 1
        mean_new = mean + (x_t - mean) / n_new
        var_new = ((n - 1) * var + (x_t - mean) * (x_t - mean_new)) / max(n, 1)

        new_means[i + 1] = mean_new
        new_vars[i + 1] = max(var_new, 1e-6)
        new_counts[i + 1] = n_new

    self.means = new_means
    self.vars = new_vars
    self.counts = new_counts

    def _compute_outputs(self):
        R = np.exp(self.log_R)
        run_lengths = np.arange(len(R))
        return {
            "cp_prob": float(R[0]),
            "run_length_mean": float(np.sum(run_lengths * R)),
            "run_length_mode": int(run_lengths[np.argmax(R)]),
            "regime_id": self.regime_id,
        }

    def _update_regime_id(self, cp_prob):
        if cp_prob > self.cp_threshold:
            self.cp_counter += 1
        else:
            self.cp_counter = 0

        if self.cp_counter >= self.cp_confirm_steps:
            self.regime_id += 1
            self.cp_counter = 0


# =====================================================
# Synthetic Data Generators
# =====================================================

def generate_case_A(T=500, cp=250):
    x1 = np.random.normal(0.0, 1.0, cp)
    x2 = np.random.normal(0.0, 2.5, T - cp)
    return np.concatenate([x1, x2]), cp

def generate_case_B(T=500, spike_prob=0.02, spike_scale=8.0):
    x = np.random.normal(0.0, 1.0, T)
    for i in range(T):
        if np.random.rand() < spike_prob:
            x[i] += np.random.choice([-1, 1]) * spike_scale
    return x

def generate_case_C(T=600):
    x = []
    for t in range(T):
        sigma = 1.0 if t < 300 else 1.0 + 0.01 * (t - 300)
        x.append(np.random.normal(0.0, sigma))
    return np.array(x)

# =====================================================
# Validation Suite
# =====================================================

def run_case(name, data, expected_cp=None):
    bocpd = BOCPD()
    cp_probs = []
    run_lengths = []

    for t, x in enumerate(data):
        out = bocpd.update(x)
        cp_probs.append(out["cp_prob"])
        run_lengths.append(out["run_length_mean"])

    print(f"\n=== {name} ===")

    if expected_cp is not None:
        peak_t = np.argmax(cp_probs)
        print(f"Expected CP ~ {expected_cp}, Detected CP ~ {peak_t}")
        assert abs(peak_t - expected_cp) < 20, "❌ CP detection is off"

    if name == "Case B":
        assert np.max(cp_probs) < 0.9, "❌ Spike triggered false CP"

    print("✔ Passed")


def run_validation_suite():
    np.random.seed(42)

    # Case A
    data_A, cp = generate_case_A()
    run_case("Case A (Clean CP)", data_A, expected_cp=cp)

    # Case B
    data_B = generate_case_B()
    run_case("Case B (Spike Robustness)", data_B)

    # Case C
    data_C = generate_case_C()
    run_case("Case C (Slow Drift)", data_C)

    print("\n🎉 All BOCPD sanity checks passed.")


if __name__ == "__main__":
    run_validation_suite()
