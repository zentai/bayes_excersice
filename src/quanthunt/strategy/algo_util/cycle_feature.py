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

cfg = CycleKalmanConfig(
    Q_base=1e-4,
    R_base=1e-3,
    use_fallback_vol=True,
    w_fallback=0.1,   # 先小小用
)

ckf = CycleKalmanOnline(cfg)

df["drift"] = rolling_logret_mean(df["close"], window=20)

out_x = []
out_K = []
out_R = []
for i in range(len(df)):
    row = df.iloc[i]
    res = ckf.update(
        close=float(row["close"]),
        drift=float(row["drift"]),
        mosaic_force_err=float(row.get("mosaic_force_err", 0.0)),
        mosaic_price_err=float(row.get("mosaic_price_err", 0.0)),
        log_vol=float(row["log_vol"]) if "log_vol" in df.columns else None,
        log_vol_global=float(row["log_vol_global"]) if "log_vol_global" in df.columns else None,
        log_volat=float(row["log_volat"]) if "log_volat" in df.columns else None,
        log_volat_global=float(row["log_volat_global"]) if "log_volat_global" in df.columns else None,
    )
    out_x.append(res["x"])
    out_K.append(res["K"])
    out_R.append(res["R_t"])

df["cycle_state_price"] = out_x
df["cycle_K"] = out_K
df["cycle_Rt"] = out_R

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CycleKalmanConfig:
    # base noise
    Q_base: float = 1e-4
    R_base: float = 1e-3

    # MOSAIC -> (Q,R) scaling strengths
    # force_err / price_err 建議用「去尺度」量（例如 z、或已正規化後的 err）
    kR_force: float = 0.35   # force_err 大 -> 更不信 Close -> R 放大
    kR_price: float = 0.25   # price_err 大 -> 更不信 Close -> R 放大
    kQ_force: float = 0.10   # force_err 大 -> state 演化更不確定 -> Q 放大（通常小一點）
    kQ_price: float = 0.08   # price_err 大 -> Q 放大（通常小一點）

    # MOSAIC error squashing (避免爆炸)
    err_clip: float = 3.0    # 例如 z-score 裁到 [-3,3]
    scale_clip_min: float = 0.2
    scale_clip_max: float = 8.0

    # optional fallback (原本 global vol / jump)
    use_fallback_vol: bool = True
    w_fallback: float = 0.15   # fallback 權重，建議小（0~0.2）
    alpha_vol: float = 0.6     # volume factor strength
    gamma_volat: float = 0.2   # volatility factor strength
    beta_jump: float = 0.2     # jump factor strength

    # init
    P0: float = 1.0
    x0: float | None = None


class CycleKalmanOnline:
    """
    1D Cycle Kalman (phase anchor) with MOSAIC-adaptive Q/R.
    State: x_t (cycle structural price)
    Obs:   y_t = close_t
    Pred:  x_pred = x_{t-1} * exp(drift_t)   (optional drift)
    """

    def __init__(self, cfg: CycleKalmanConfig):
        self.cfg = cfg
        self.x = cfg.x0
        self.P = cfg.P0
        self.prev_close = None

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return float(np.minimum(np.maximum(x, lo), hi))

    def _mosaic_scale(self, force_err: float, price_err: float) -> tuple[float, float]:
        """
        Convert MOSAIC errors into multiplicative scales for (Q, R).
        errors should be roughly standardized (e.g., z-like); we squash & clip.
        """
        c = self.cfg

        fe = self._clip(force_err, -c.err_clip, c.err_clip)
        pe = self._clip(price_err, -c.err_clip, c.err_clip)

        # use magnitude only (避免 direction 把 HMM/KF 定義成極端值方向)
        fe = abs(fe)
        pe = abs(pe)

        # scale = exp(sum(k * err))
        R_scale = np.exp(c.kR_force * fe + c.kR_price * pe)
        Q_scale = np.exp(c.kQ_force * fe + c.kQ_price * pe)

        R_scale = self._clip(R_scale, c.scale_clip_min, c.scale_clip_max)
        Q_scale = self._clip(Q_scale, c.scale_clip_min, c.scale_clip_max)
        return float(Q_scale), float(R_scale)

    def _fallback_scale(self, close: float, log_vol: float, log_vol_global: float,
                        log_volat: float, log_volat_global: float) -> float:
        """
        Your old-style adaptive R factor (volume/volatility/jump), compressed.
        Return a multiplicative factor >= 0.
        """
        c = self.cfg
        if self.prev_close is None:
            jump = 0.0
        else:
            jump = abs(close - self.prev_close)

        # Similar spirit as your prior code (but bounded softly)
        vol_factor = np.exp(-c.alpha_vol * (log_vol - log_vol_global))
        volat_factor = np.exp(c.gamma_volat * (log_volat - log_volat_global))
        jump_factor = np.exp(c.beta_jump * jump)

        raw = vol_factor * volat_factor * jump_factor
        # clip to avoid insane scaling
        return self._clip(float(raw), 0.2, 8.0)

    def update(self,
               close: float,
               drift: float = 0.0,
               mosaic_force_err: float = 0.0,
               mosaic_price_err: float = 0.0,
               # fallback inputs (optional)
               log_vol: float | None = None,
               log_vol_global: float | None = None,
               log_volat: float | None = None,
               log_volat_global: float | None = None) -> dict:
        """
        One-step online update.
        drift: e.g. rolling mean log-return (can be 0 if you want pure RW)
        mosaic_force_err / mosaic_price_err: standardized errors (z-like)
        """
        c = self.cfg

        # init x
        if self.x is None:
            self.x = close
            self.prev_close = close
            return {
                "x": self.x, "P": self.P,
                "K": 0.0, "Q_t": c.Q_base, "R_t": c.R_base,
                "x_pred": self.x, "resid": 0.0
            }

        # ---- Prediction (cycle anchor with drift) ----
        x_pred = float(self.x * np.exp(drift))
        P_pred = float(self.P + c.Q_base)  # base; will scale with Q_t

        # ---- Adaptive Q/R from MOSAIC ----
        Q_scale, R_scale = self._mosaic_scale(mosaic_force_err, mosaic_price_err)

        Q_t = float(c.Q_base * Q_scale)
        R_t = float(c.R_base * R_scale)

        # apply Q_t into P_pred
        P_pred = float(self.P + Q_t)

        # ---- Optional fallback: keep tiny influence of old vol logic ----
        if c.use_fallback_vol and (log_vol is not None) and (log_vol_global is not None) \
           and (log_volat is not None) and (log_volat_global is not None):
            fb = self._fallback_scale(close, log_vol, log_vol_global, log_volat, log_volat_global)
            # blend multiplicatively (light touch)
            R_t *= float((1.0 - c.w_fallback) + c.w_fallback * fb)

        # ---- Update ----
        resid = float(close - x_pred)
        S = float(P_pred + R_t)            # innovation variance
        K = float(P_pred / S)              # Kalman gain

        self.x = float(x_pred + K * resid)
        self.P = float((1.0 - K) * P_pred)

        self.prev_close = close

        return {
            "x": self.x,
            "P": self.P,
            "K": K,
            "Q_t": Q_t,
            "R_t": R_t,
            "x_pred": x_pred,
            "resid": resid,
            "S": S,
        }


def rolling_logret_mean(close: pd.Series, window: int = 20) -> pd.Series:
    lr = np.log(close).diff()
    return lr.rolling(window).mean().fillna(0.0)

def main():
    cycle_hmm = CycleHMM(
        n_states=4, atr_window=60, bayes_window=20, kalman_params={...}
    )

    cycle_hmm.partial_fit(df_chunk)  # online 更新
    state = cycle_hmm.current_state  # 目前 phase
    prob = cycle_hmm.state_prob  # state posterior
    signal = cycle_hmm.phase_signal  # 是否允許「週期型進場」

