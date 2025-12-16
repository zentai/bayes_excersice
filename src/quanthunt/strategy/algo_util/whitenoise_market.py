import numpy as np
import pandas as pd


def generate_true_force(T=2000, seed=42):
    """
    0=quiet, 1=trend, 2=chaos
    特色：
    - regime 一定會切
    - trend slope 會漂移
    - 每段結束會「崩壞或反轉」（避免永遠單調趨勢宇宙）
    """
    rng = np.random.default_rng(seed)

    force = np.zeros(T)
    force_trend = np.zeros(T)
    regimes = np.zeros(T)

    slope = 0.0
    t = 0

    while t < T:
        r = rng.choice([0, 1, 2], p=[0.35, 0.40, 0.25])
        length = int(rng.integers(80, 240))
        end = min(T, t + length)
        regimes[t:end] = r

        for i in range(t, end):
            if r == 0:  # quiet
                slope = 0.7 * slope + rng.normal(0, 0.01)
                force[i] = (force[i - 1] if i > 0 else 0.0) + rng.normal(0, 0.25)

            elif r == 1:  # trend
                # slope 是 stochastic（會漂移）
                slope = 0.92 * slope + rng.normal(0, 0.02)
                force[i] = (
                    (force[i - 1] if i > 0 else 0.0) + slope + rng.normal(0, 0.20)
                )

            else:  # chaos
                slope = 0.5 * slope + rng.normal(0, 0.08)
                force[i] = (force[i - 1] if i > 0 else 0.0) + rng.normal(0, 0.9)

            force_trend[i] = slope

        # ✅ 每段結束做「結構破壞」：趨勢會失效或反轉
        if r == 1:
            if rng.random() < 0.5:
                slope *= -0.5  # 反向 + 衰減
            else:
                slope *= 0.2  # 崩壞（失效）

        t = end

    # 讓 force 尺度不要無限漂走（避免 price 二次爆炸）
    force = (force - force.mean()) / (force.std() + 1e-9)

    return force, force_trend, regimes


def generate_price_from_force(force, seed=123):
    """
    price dynamics:
      accel ~ force
      speed = ∫ accel
      price = ∫ speed
    """
    rng = np.random.default_rng(seed)

    T = len(force)
    accel = 0.05 * force + rng.normal(0, 0.02, T)
    speed = np.cumsum(accel)
    price = np.cumsum(speed)

    # 加一点 observation noise
    price_obs = price + rng.normal(0, 0.5, T)

    return price_obs, price, speed, accel


def generate_force_proxy(force, noise_scale=0.8, seed=7):
    """
    模拟 pseudo_delta / HL_ret 这类 proxy：
    强相关，但噪音很大
    """
    rng = np.random.default_rng(seed)
    proxy = force + rng.normal(0, noise_scale, len(force))
    return proxy


def generate_latent_price_from_force(force, seed=123):
    rng = np.random.default_rng(seed)
    T = len(force)

    accel = np.zeros(T)
    speed = np.zeros(T)
    price = np.zeros(T)

    for t in range(1, T):
        noise = rng.normal(0, 0.02)

        # ✅ 关键修改在这里：
        # accel 有惯性、有衰减，而不是直接 = force
        accel[t] = (
            0.85 * accel[t - 1]  # inertia / fatigue
            + 0.05 * force[t]  # force influence
            + noise
        )

        speed[t] = speed[t - 1] + accel[t]
        price[t] = price[t - 1] + speed[t]

    return price, speed, accel


def project_to_ohlcv(latent_price, force, regimes, seed=999):
    rng = np.random.default_rng(seed)
    T = len(latent_price)

    Open = np.zeros(T)
    High = np.zeros(T)
    Low = np.zeros(T)
    Close = np.zeros(T)
    Volume = np.zeros(T)

    Close[:] = latent_price + rng.normal(0, 0.3, T)

    Open[0] = Close[0]
    Open[1:] = Close[:-1] + rng.normal(0, 0.1, T - 1)

    for t in range(T):
        # intrabar volatility depends on regime + force magnitude
        base_range = 0.5 + 0.5 * abs(force[t])
        regime_amp = {0: 0.5, 1: 1.0, 2: 2.0}[int(regimes[t])]
        bar_range = base_range * regime_amp

        wick_up = abs(rng.normal(0, bar_range))
        wick_dn = abs(rng.normal(0, bar_range))

        High[t] = max(Open[t], Close[t]) + wick_up
        Low[t] = min(Open[t], Close[t]) - wick_dn

        # volume also responds to force & regime
        Volume[t] = 100 + 50 * abs(force[t]) + 80 * regime_amp + rng.normal(0, 10)

    return pd.DataFrame(
        {
            "Open": Open,
            "High": High,
            "Low": Low,
            "Close": Close,
            "Vol": Volume,
        }
    )


def build_synthetic_market_ohlcv(T=3000, seed=42):
    true_force, true_force_trend, regimes = generate_true_force(T, seed=seed)
    latent_price, true_speed, true_accel = generate_latent_price_from_force(
        true_force, seed=seed + 1
    )

    ohlcv = project_to_ohlcv(latent_price, true_force, regimes, seed=seed + 2)

    # === 真相（只用于验证，不喂模型）===
    ohlcv["true_force"] = true_force
    ohlcv["true_force_trend"] = true_force_trend
    ohlcv["true_price"] = latent_price
    ohlcv["true_speed"] = true_speed
    ohlcv["true_accel"] = true_accel
    ohlcv["true_regime"] = regimes

    return ohlcv
