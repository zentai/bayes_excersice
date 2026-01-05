import numpy as np
import pandas as pd


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


import numpy as np
from dataclasses import dataclass


# -------------------------
# Config
# -------------------------
@dataclass
class CycleKalmanConfig:
    Q_base: float = 1e-4
    R_base: float = 1e-3

    # R reacts FAST to instantaneous MOSAIC error
    kR: float = 0.45  # strength for R scaling by |err|
    err_clip: float = 3.0  # clip mosaic_err (z-like) to avoid explosions
    R_min_scale: float = 0.2
    R_max_scale: float = 12.0

    # Q reacts SLOW to sustained error (EWMA / accumulator)
    q_mode: str = "ewma"  # "ewma" or "accum"
    kQ: float = 0.30  # strength for Q scaling by sustained error
    q_floor: float = 0.0  # baseline sustained error floor
    Q_min_scale: float = 0.5
    Q_max_scale: float = 6.0

    # sustained-error dynamics
    ewma_alpha: float = 0.06  # smaller = slower
    accum_beta: float = 0.02  # leak rate for accumulator
    accum_gain: float = 0.10  # how fast it accumulates on big error
    q_trigger: float = 1.2  # only start increasing Q if sustained err > trigger

    # optional drift (cycle stack)
    P0: float = 1.0
    x0: float | None = None


# -------------------------
# Online Cycle Kalman
# -------------------------
class CycleKalmanOnline:
    """
    1D Cycle Kalman:
      state x_t: cycle structural price (phase anchor)
      obs   y_t: close_t

    Key design:
      - R_t reacts quickly to instantaneous MOSAIC error -> "don't trust close now"
      - Q_t reacts slowly to sustained MOSAIC error      -> "world model may be less reliable"
    """

    def __init__(self, cfg: CycleKalmanConfig):
        self.cfg = cfg
        self.x = cfg.x0
        self.P = cfg.P0

        # sustained error trackers
        self.err_ewma = 0.0
        self.err_accum = 0.0

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return float(np.minimum(np.maximum(x, lo), hi))

    def _instant_err(self, mosaic_err: float) -> float:
        c = self.cfg
        e = self._clip(mosaic_err, -c.err_clip, c.err_clip)
        return abs(e)

    def _update_sustained(self, e_abs: float) -> float:
        """
        Returns sustained error signal (>=0) used to drive Q slowly.
        """
        c = self.cfg

        if c.q_mode == "ewma":
            self.err_ewma = (1.0 - c.ewma_alpha) * self.err_ewma + c.ewma_alpha * e_abs
            return float(self.err_ewma)

        if c.q_mode == "accum":
            # leaky accumulator: grows when error large, leaks otherwise
            self.err_accum = (1.0 - c.accum_beta) * self.err_accum + c.accum_gain * max(
                0.0, e_abs - c.q_floor
            )
            return float(self.err_accum)

        raise ValueError("q_mode must be 'ewma' or 'accum'")

    def _scale_R(self, e_abs: float) -> float:
        c = self.cfg
        # fast, strong response
        scale = np.exp(c.kR * e_abs)
        return self._clip(float(scale), c.R_min_scale, c.R_max_scale)

    def _scale_Q(self, e_sust: float) -> float:
        c = self.cfg
        # slow + gated response: only increase Q when sustained error is high enough
        if e_sust <= c.q_trigger:
            return 1.0

        scale = np.exp(c.kQ * (e_sust - c.q_trigger))
        return self._clip(float(scale), c.Q_min_scale, c.Q_max_scale)

    def update(self, close: float, drift: float = 0.0, mosaic_err: float = 0.0) -> dict:
        """
        One online step.
        drift: e.g. rolling mean log-return (can be 0.0)
        mosaic_err: a single MOSAIC "structure error" scalar (z-like preferred),
                    e.g. abs(z_mix) or combined force/price error.
        """
        c = self.cfg

        # init
        if self.x is None:
            self.x = float(close)
            return {
                "x": self.x,
                "P": self.P,
                "K": 0.0,
                "Q_t": c.Q_base,
                "R_t": c.R_base,
                "x_pred": self.x,
                "resid": 0.0,
                "e_abs": 0.0,
                "e_sust": 0.0,
            }

        # ---- prediction (cycle stack) ----
        x_pred = float(self.x * np.exp(drift))

        # ---- MOSAIC error -> instantaneous & sustained ----
        e_abs = self._instant_err(mosaic_err)
        e_sust = self._update_sustained(e_abs)

        # ---- adaptive R (fast) & Q (slow) ----
        R_t = float(c.R_base * self._scale_R(e_abs))
        Q_t = float(c.Q_base * self._scale_Q(e_sust))

        # ---- Kalman update ----
        P_pred = float(self.P + Q_t)
        resid = float(close - x_pred)
        S = float(P_pred + R_t)
        K = float(P_pred / S)

        self.x = float(x_pred + K * resid)
        self.P = float((1.0 - K) * P_pred)

        return {
            "x": self.x,
            "P": self.P,
            "K": K,
            "Q_t": Q_t,
            "R_t": R_t,
            "x_pred": x_pred,
            "resid": resid,
            "S": S,
            "e_abs": e_abs,
            "e_sust": e_sust,
        }


# -------------------------
# Example usage (minimal)
# -------------------------
if __name__ == "__main__":
    import pandas as pd

    # Suppose df has: close, drift, mosaic_err
    df = pd.DataFrame(
        {
            "Close": [100, 101, 105, 104, 103, 120, 119, 118],
            "drift": [0, 0.001, 0.001, 0, 0, 0.0, 0.0, 0.0],
            # mosaic_err (z-like): spikes when structure looks wrong
            "mosaic_err": [0.2, 0.3, 2.0, 1.5, 0.8, 3.5, 2.8, 1.0],
        }
    )

    cfg = CycleKalmanConfig(q_mode="ewma")
    ckf = CycleKalmanOnline(cfg)

    xs, Ks, Qs, Rs = [], [], [], []
    for _, r in df.iterrows():
        out = ckf.update(float(r["Close"]), float(r["drift"]), float(r["mosaic_err"]))
        xs.append(out["x"])
        Ks.append(out["K"])
        Qs.append(out["Q_t"])
        Rs.append(out["R_t"])

    df["cycle_state"] = xs
    df["K"] = Ks
    df["Q_t"] = Qs
    df["R_t"] = Rs

    print(df)


def rolling_logret_mean(close: pd.Series, window: int = 20) -> pd.Series:
    lr = np.log(close).diff()
    return lr.rolling(window).mean().fillna(0.0)


def main():
    cfg = CycleKalmanConfig(
        Q_base=1e-4,
        R_base=1e-3,
        use_fallback_vol=True,
        w_fallback=0.1,  # 先小小用
    )

    ckf = CycleKalmanOnline(cfg)

    df["drift"] = rolling_logret_mean(df["Close"], window=20)

    out_x = []
    out_K = []
    out_R = []
    for i in range(len(df)):
        row = df.iloc[i]
        res = ckf.update(
            close=float(row["Close"]),
            drift=float(row["drift"]),
            mosaic_force_err=float(row.get("mosaic_force_err", 0.0)),
            mosaic_price_err=float(row.get("mosaic_price_err", 0.0)),
            log_vol=float(row["log_vol"]) if "log_vol" in df.columns else None,
            log_vol_global=(
                float(row["log_vol_global"]) if "log_vol_global" in df.columns else None
            ),
            log_volat=float(row["log_volat"]) if "log_volat" in df.columns else None,
            log_volat_global=(
                float(row["log_volat_global"])
                if "log_volat_global" in df.columns
                else None
            ),
        )
        out_x.append(res["x"])
        out_K.append(res["K"])
        out_R.append(res["R_t"])

    df["cycle_state_price"] = out_x
    df["cycle_K"] = out_K
    df["cycle_Rt"] = out_R

    # cycle_hmm = CycleHMM(
    #     n_states=4, atr_window=60, bayes_window=20, kalman_params={...}
    # )

    # cycle_hmm.partial_fit(df_chunk)  # online 更新
    # state = cycle_hmm.current_state  # 目前 phase
    # prob = cycle_hmm.state_prob  # state posterior
    # signal = cycle_hmm.phase_signal  # 是否允許「週期型進場」
