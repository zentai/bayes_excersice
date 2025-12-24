import pandas as pd
import numpy as np


def sortino_ratio(series, rf=0.0):
    ZERO = 1e-9

    downside = series[series < rf]
    downside_std = downside.std(ddof=0)

    if np.isnan(downside_std) or downside_std == 0:
        downside_std = ZERO

    return (series.mean() - rf) / downside_std


def normalize_hmm_features(series, window=20):
    ZERO = 1e-9

    _log = np.log(series + ZERO)
    _ma = series.rolling(window).mean()
    _relative_strength = series / _ma
    _zscore = (series - _ma) / series.rolling(window).std()
    _sortino = sortino_ratio(series)

    return _log, _ma, _relative_strength, _zscore, _sortino


def performance_metrics(series, rf=0.0):
    """
    Calculate common performance metrics from a profit series.
    """

    # Simulate cash flow growth
    initial_cash = 1000.0
    cash = initial_cash
    for profit in series[series.notna()]:
        cash += 100 * profit
        if cash <= 0:
            cash = 0
            break

    mean = series.mean()
    median = series.median()
    std = series.std(ddof=0)
    sharpe = (mean - rf) / std if std > 0 else np.nan
    sortino = sortino_ratio(series, rf)

    cumulative = series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    volatility = std
    win_rate = (series > 0).mean()

    drawdown_streak = 0
    max_drawdown_streak = 0
    for r in series:
        if r < 0:
            drawdown_streak += 1
            max_drawdown_streak = max(max_drawdown_streak, drawdown_streak)
        else:
            drawdown_streak = 0

    return {
        "Sample": len(series),
        "mean": mean,
        "median": median,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "max_drawdown_streak": max_drawdown_streak,
        "volatility": volatility,
        "win_rate": win_rate,
        "final_cash": cash,
    }


def compare_signal_filters(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare performance under different signal filters:
    - No filter
    - BuySignal only
    - HMM only
    - BuySignal + HMM
    """

    results = []

    # --- 0) No filter ---
    metrics = performance_metrics(base_df.profit)
    metrics["name"] = "No Filter"
    metrics["R-multiple (min)"] = base_df["P/L"].min()
    metrics["R-multiple (mean)"] = base_df["P/L"].mean()
    metrics["R-multiple (max)"] = base_df["P/L"].max()
    results.append(metrics)

    # --- 1) BuySignal only ---
    mask = base_df.BuySignal == True
    metrics = performance_metrics(base_df[mask].profit)
    metrics["name"] = "BuySignal only"
    metrics["R-multiple (min)"] = base_df[mask]["P/L"].min()
    metrics["R-multiple (mean)"] = base_df[mask]["P/L"].mean()
    metrics["R-multiple (max)"] = base_df[mask]["P/L"].max()
    results.append(metrics)

    # --- 2) HMM only (diagnostic, not tradable) ---
    mask = base_df.HMM_Signal == 1
    metrics = performance_metrics(base_df[mask].profit)
    metrics["name"] = "HMM only"
    metrics["R-multiple (min)"] = base_df[mask]["P/L"].min()
    metrics["R-multiple (mean)"] = base_df[mask]["P/L"].mean()
    metrics["R-multiple (max)"] = base_df[mask]["P/L"].max()
    results.append(metrics)

    # --- 3) BuySignal + HMM ---
    mask = (base_df.BuySignal == True) & (base_df.HMM_Signal == 1)
    metrics = performance_metrics(base_df[mask].profit)
    metrics["name"] = "BuySignal + HMM"
    metrics["R-multiple (min)"] = base_df[mask]["P/L"].min()
    metrics["R-multiple (mean)"] = base_df[mask]["P/L"].mean()
    metrics["R-multiple (max)"] = base_df[mask]["P/L"].max()
    results.append(metrics)

    df = pd.DataFrame(results).set_index("name")
    return df


def analyze_hmm_states(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance by individual HMM_State.
    """

    results = []

    for state in sorted(base_df.HMM_State.dropna().unique()):
        mask = base_df.HMM_State == state
        if mask.sum() < 10:
            continue  # 樣本太少直接跳過

        metrics = performance_metrics(base_df[mask].profit)
        metrics["HMM_State"] = int(state)
        metrics["sample"] = mask.sum()
        metrics["R-multiple (min)"] = base_df[mask]["P/L"].min()
        metrics["R-multiple (mean)"] = base_df[mask]["P/L"].mean()
        metrics["R-multiple (max)"] = base_df[mask]["P/L"].max()
        results.append(metrics)

    return pd.DataFrame(results).set_index("HMM_State")
