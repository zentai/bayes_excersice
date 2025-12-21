class CycleFeatureExtractor:
    def __init__(self, atr_window=60, bayes_window=20):
        self.atr_window = atr_window
        self.bayes_window = bayes_window
        self.ZERO = 1e-9

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # --- Kalman price (外部傳入或已算好) ---
        # df["Kalman"], df["ATR"], df["ema_long"] 預期已存在

        # === 結構偏離（週期核心）===
        df["trend"] = df.Kalman - df.ema_long
        df["trend_norm"] = df.trend / (df.ATR + self.ZERO)

        df["bias_off_norm"] = (df.Kalman - df.Close).abs() / (df.ATR + self.ZERO)

        df["TR_norm"] = np.log(df.ATR + self.ZERO) / np.log(
            df.ATR.rolling(self.atr_window).mean() + self.ZERO
        )

        # === 行為 / phase 訊號 ===
        df["ret"] = np.log(df.Close / df.Close.shift()).replace(
            [np.inf, -np.inf], np.nan
        )

        # up / down streak（phase 轉折很好用）
        up, down = [], []
        cu, cd = 0, 0
        for r in df["ret"].fillna(0).values:
            if r > 0:
                cu += 1
                cd = 0
            elif r < 0:
                cd += 1
                cu = 0
            else:
                cu = cd = 0
            up.append(cu)
            down.append(cd)

        df["up_streak_norm"] = np.array(up) / max(self.atr_window, 1)
        df["down_streak_norm"] = np.array(down) / max(self.atr_window, 1)

        # skew（過熱 / 過冷 proxy）
        df["skew_ret"] = df["ret"].rolling(self.atr_window, min_periods=10).skew()

        # drawdown proxy（週期頂底）
        roll_max = df.Close.cummax()
        df["dd_proxy"] = (roll_max - df.Close) / (roll_max + self.ZERO)

        features = [
            "trend_norm",
            "bias_off_norm",
            "TR_norm",
            "up_streak_norm",
            "down_streak_norm",
            "skew_ret",
            "dd_proxy",
        ]

        df[features] = df[features].fillna(method="bfill").fillna(method="ffill")
        return df[features].values


from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class CycleHMM:
    def __init__(self, n_states=4, atr_window=60, bayes_window=20):
        self.n_states = n_states
        self.extractor = CycleFeatureExtractor(atr_window, bayes_window)
        self.scaler = StandardScaler()

        self.hmm = GaussianHMM(
            n_components=n_states, covariance_type="full", n_iter=10, warm_start=True
        )

        self._fitted = False
        self.current_state = None
        self.state_prob = None

    def partial_fit(self, df: pd.DataFrame):
        X = self.extractor.transform(df)

        Xs = (
            self.scaler.fit_transform(X)
            if not self._fitted
            else self.scaler.transform(X)
        )

        if not self._fitted:
            self.hmm.fit(Xs)
            self._fitted = True
        else:
            self.hmm.fit(Xs)

        # online 狀態
        self.state_prob = self.hmm.predict_proba(Xs)[-1]
        self.current_state = int(np.argmax(self.state_prob))

    @property
    def phase_signal(self) -> int:
        """
        週期型進場允許信號
        （例如只在回升 phase）
        """
        # 假設 state 1 = 回升期（需你回測確認語義）
        return 1 if self.current_state == 1 else 0


def main():
    cycle_hmm = CycleHMM(
        n_states=4, atr_window=60, bayes_window=20, kalman_params={...}
    )

    cycle_hmm.partial_fit(df_chunk)  # online 更新
    state = cycle_hmm.current_state  # 目前 phase
    prob = cycle_hmm.state_prob  # state posterior
    signal = cycle_hmm.phase_signal  # 是否允許「週期型進場」
