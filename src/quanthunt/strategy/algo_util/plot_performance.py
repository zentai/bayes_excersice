"""
strategy_utils.py

工具集合：
1) 核心：樣本量檢查（勝率估計是否有統計意義）
2) 策略比較圖：
   - KDE 分布比較
   - CDF 分布比較
   - KL Divergence 矩陣
   - 左尾風險 vs 右尾獲利 象限圖
"""

from math import ceil
from typing import Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, entropy


# ================================
#  样本量檢查（勝率估計）
# ================================


def check_sample_size(
    profits: Iterable[float],
    epsilon: float = 0.1,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    檢查「勝率估計」的樣本量是否足夠。

    參數
    ----
    profits : 可迭代的單筆交易盈虧（>0 視為贏，<=0 視為輸）
    epsilon : 勝率估計允許的誤差範圍（例如 0.02 = ±2%）
    confidence : 信心水準（例如 0.95 = 95%）

    回傳
    ----
    {
        "winrate": float or None,
        "required_n": int or None,
        "total_n": int,
        "enough": bool,
        "status": str,
    }
    """
    profits = np.asarray(list(profits), dtype=float)
    total = profits.size

    if total == 0:
        return {
            "winrate": None,
            "required_n": None,
            "total_n": 0,
            "enough": False,
            "status": "⚠️ 無樣本資料（profits 為空）",
        }

    wins = np.sum(profits > 0)
    winrate = wins / total

    # z-score 對應信心水準
    z = norm.ppf((1.0 + confidence) / 2.0)

    # 需要的樣本數（比例估計公式）
    required_n = int(ceil((z**2 * winrate * (1.0 - winrate)) / (epsilon**2)))

    if total < required_n:
        status = (
            f"⚠️ 樣本不足：目前 {total} 筆，需要 ≥ {required_n} 筆 "
            f"才能以 ±{epsilon*100:.1f}% 精度估計勝率（{confidence*100:.0f}% 信心）"
        )
        enough = False
    else:
        status = (
            f"✅ 樣本達標：目前 {total} 筆，已超過所需 {required_n} 筆，"
            f"勝率估計在 ±{epsilon*100:.1f}%、{confidence*100:.0f}% 信心水準下具統計參考性"
        )
        enough = True

    return {
        "winrate": float(winrate),
        "required_n": required_n,
        "total_n": total,
        "enough": enough,
        "status": status,
    }


# ================================
#  策略比較圖：KDE
# ================================


def plot_strategy_kde_comparison(
    strategy_dict: Dict[str, Iterable[float]],
    title: str = "Profit Distribution Comparison (KDE)",
):
    """
    比較多個策略的 profit 分布（KDE）。

    strategy_dict:
        {
            "StrategyA": profits_array,
            "StrategyB": profits_array,
            ...
        }
    """
    plt.figure(figsize=(12, 6))

    for name, profits in strategy_dict.items():
        profits = np.asarray(list(profits), dtype=float)
        if profits.size == 0:
            continue
        sns.kdeplot(profits, label=name, shade=False, linewidth=2)

    plt.axvline(0.0, color="black", linestyle="--", alpha=0.5)
    plt.title(title)
    plt.xlabel("Profit")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# ================================
#  策略比較圖：CDF
# ================================


def plot_strategy_cdf_comparison(
    strategy_dict: Dict[str, Iterable[float]],
    title: str = "Profit Distribution Comparison (CDF)",
):
    """
    比較多個策略的 profit CDF（累積分布函數）。
    """
    plt.figure(figsize=(12, 6))

    for name, profits in strategy_dict.items():
        profits = np.asarray(list(profits), dtype=float)
        if profits.size == 0:
            continue
        sorted_vals = np.sort(profits)
        y = np.linspace(0.0, 1.0, sorted_vals.size)
        plt.plot(sorted_vals, y, label=name, linewidth=2)

    plt.axvline(0.0, color="black", linestyle="--", alpha=0.5)
    plt.title(title)
    plt.xlabel("Profit")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# ================================
#  KL Divergence 矩陣
# ================================


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p + eps
    q = q + eps
    # scipy.stats.entropy = KL(p || q)
    return float(entropy(p, q))


def plot_kl_matrix(
    strategy_dict: Dict[str, Iterable[float]],
    bins: int = 50,
    title: str = "KL Divergence Between Strategy Profit Distributions",
):
    """
    對多個策略的 profit 分布做 KL Divergence 矩陣並畫 heatmap。
    KL 越小 → 分布越相似；越大 → 越不一樣。
    """
    names = list(strategy_dict.keys())
    if not names:
        raise ValueError("strategy_dict 為空")

    # 準備所有 profits
    arrays = []
    for name in names:
        arr = np.asarray(list(strategy_dict[name]), dtype=float)
        if arr.size == 0:
            raise ValueError(f"策略 {name} 的 profits 為空，無法計算 KL")
        arrays.append(arr)

    all_values = np.hstack(arrays)
    hist_range = (np.min(all_values), np.max(all_values))

    hist_dict = {}
    for name, arr in zip(names, arrays):
        hist, _ = np.histogram(arr, bins=bins, range=hist_range, density=True)
        hist_dict[name] = hist

    n = len(names)
    mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            mat[i, j] = _kl(hist_dict[names[i]], hist_dict[names[j]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        xticklabels=names,
        yticklabels=names,
        cmap="viridis",
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ================================
#  左尾風險 vs 右尾獲利 象限圖
# ================================


# -------------------------
#  Tail Metrics (CVaR)
# -------------------------


def cvar_left_tail(profits, alpha=0.05):
    """更真實的 Left Tail Risk（平均最差 5%）"""
    var = np.percentile(profits, alpha * 100)
    return profits[profits <= var].mean()


def cvar_right_tail(profits, alpha=0.95):
    """更真實的 Right Tail Gain（平均最好 5%）"""
    var = np.percentile(profits, alpha * 100)
    return profits[profits >= var].mean()


# -------------------------
#  Quadrant Performance Plot
# -------------------------


def plot_tail_quadrant(strategy_dict):
    """
    strategy_dict = {
        "base": profits_array,
        "Kalman_LSA_kf": profits_array,
        ...
    }
    """
    plt.figure(figsize=(8, 8))

    for name, profits in strategy_dict.items():
        lt = abs(cvar_left_tail(profits))  # 正化（越大越代表虧得深）
        rt = cvar_right_tail(profits)  # 越大越好

        plt.scatter(lt, rt, s=120, label=name)
        plt.text(lt, rt, name)

    plt.axvline(0, color="gray", linestyle="--")
    plt.axhline(0, color="gray", linestyle="--")

    plt.xlabel("Left Tail Risk (CVaR95, larger = deeper loss)")
    plt.ylabel("Right Tail Gain (CVaR95, larger = better)")
    plt.title("Real Tail Risk–Reward Quadrant")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


__all__ = [
    "check_sample_size",
    "plot_strategy_kde_comparison",
    "plot_strategy_cdf_comparison",
    "plot_kl_matrix",
    "plot_tail_quadrant",
]
