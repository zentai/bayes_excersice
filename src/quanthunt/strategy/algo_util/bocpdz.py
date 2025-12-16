import numpy as np

# =====================================================
# BOCPD (same as your current Gaussian online version)
# =====================================================

class BOCPD:
    def __init__(
        self,
        hazard=0.01,
        r_max=300,
        prune_threshold=1e-6,
        cp_threshold=0.5,
        cp_confirm_steps=3,
    ):
        self.hazard = hazard
        self.r_max = r_max
        self.prune_threshold = prune_threshold
        self.cp_threshold = cp_threshold
        self.cp_confirm_steps = cp_confirm_steps

        self.log_R = np.array([0.0])
        self.means = np.array([0.0])
        self.vars = np.array([1.0])
        self.counts = np.array([1])

        self.cp_counter = 0
        self.regime_id = 0

    def update(self, x):
        log_pred = self._log_likelihood(x)
        self._update_runlength(log_pred)
        self._expand_stats(x)
        self._normalize_and_prune()
        out = self._outputs()
        self._update_regime(out["cp_prob"])
        return out

    def _log_likelihood(self, x):
        var = self.vars + 1e-8
        return -0.5 * (np.log(2 * np.pi * var) + (x - self.means) ** 2 / var)

    def _update_runlength(self, log_pred):
        log_R_new = np.full(len(self.log_R) + 1, -np.inf)
        log_R_new[1:] = self.log_R + log_pred + np.log(1 - self.hazard)
        log_R_new[0] = np.logaddexp.reduce(
            self.log_R + log_pred + np.log(self.hazard)
        )
        self.log_R = log_R_new

    def _expand_stats(self, x):
        new_means = np.empty(len(self.log_R))
        new_vars = np.empty(len(self.log_R))
        new_counts = np.empty(len(self.log_R))

        new_means[0] = x
        new_vars[0] = 1.0
        new_counts[0] = 1

        for i in range(len(self.means)):
            n = self.counts[i]
            mu = self.means[i]
            var = self.vars[i]

            n2 = n + 1
            mu2 = mu + (x - mu) / n2
            var2 = ((n - 1) * var + (x - mu) * (x - mu2)) / max(n, 1)

            new_means[i + 1] = mu2
            new_vars[i + 1] = max(var2, 1e-6)
            new_counts[i + 1] = n2

        self.means = new_means
        self.vars = new_vars
        self.counts = new_counts

    def _normalize_and_prune(self):
        self.log_R -= np.logaddexp.reduce(self.log_R)
        R = np.exp(self.log_R)

        mask = R > self.prune_threshold
        self.log_R = self.log_R[mask]
        self.means = self.means[mask]
        self.vars = self.vars[mask]
        self.counts = self.counts[mask]

        if len(self.log_R) > self.r_max:
            self.log_R = self.log_R[: self.r_max]
            self.means = self.means[: self.r_max]
            self.vars = self.vars[: self.r_max]
            self.counts = self.counts[: self.r_max]

    def _outputs(self):
        R = np.exp(self.log_R)
        rl = np.arange(len(R))
        return {
            "cp_prob": R[0],
            "run_length_mean": np.sum(rl * R),
            "run_length_mode": int(rl[np.argmax(R)]),
            "regime_id": self.regime_id,
        }

    def _update_regime(self, cp_prob):
        if cp_prob > self.cp_threshold:
            self.cp_counter += 1
        else:
            self.cp_counter = 0

        if self.cp_counter >= self.cp_confirm_steps:
            self.regime_id += 1
            self.cp_counter = 0


# =====================================================
# Synthetic data generators
# =====================================================

def case_A(T=500, cp=250):
    return np.concatenate([
        np.random.normal(0, 1.0, cp),
        np.random.normal(0, 2.5, T - cp)
    ]), cp

def case_B(T=500):
    x = np.random.normal(0, 1.0, T)
    for i in range(T):
        if np.random.rand() < 0.02:
            x[i] += np.random.choice([-1, 1]) * 8
    return x

def case_C(T=600):
    x = []
    for t in range(T):
        sigma = 1.0 if t < 300 else 1.0 + 0.01 * (t - 300)
        x.append(np.random.normal(0, sigma))
    return np.array(x)


# =====================================================
# Validation logic (semantic-correct)
# =====================================================

def run_case_A():
    x, cp = case_A()
    bocpd = BOCPD()
    cp_probs = []

    for v in x:
        cp_probs.append(bocpd.update(v)["cp_prob"])

    window = cp_probs[cp: cp + 200]
    assert max(window) > 0.4, "❌ Case A: CP never confirmed after change"
    print("✔ Case A passed (confirmed structural change)")

def run_case_B():
    x = case_B()
    bocpd = BOCPD()
    cp_probs = []

    for v in x:
        cp_probs.append(bocpd.update(v)["cp_prob"])

    assert max(cp_probs) < 0.8, "❌ Case B: spike triggered false CP"
    print("✔ Case B passed (robust to spikes)")

def run_case_C():
    x = case_C()
    bocpd = BOCPD()
    cp_probs = []

    for v in x:
        cp_probs.append(bocpd.update(v)["cp_prob"])

    early = np.mean(cp_probs[200:300])
    late = np.mean(cp_probs[400:500])

    assert late > early, "❌ Case C: no sensitivity to slow drift"
    assert late < 0.9, "❌ Case C: drift misclassified as hard CP"
    print("✔ Case C passed (slow drift handled correctly)")


# =====================================================
# Entry point
# =====================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("Running BOCPD semantic validation...\n")
    run_case_A()
    run_case_B()
    run_case_C()
    print("\n🎉 All semantic BOCPD tests passed.")
