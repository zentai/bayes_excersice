import itertools
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


# ---------- 基礎工具 ----------


def longest_win_run(p: np.ndarray) -> int:
    """連續正報酬的最長長度 L（趨勢持續性指標）"""
    L = cur = 0
    for x in p:
        if x > 0:
            cur += 1
            L = max(L, cur)
        else:
            cur = 0
    return L


def compute_trend_mu(
    df_sub: pd.DataFrame,
    profit_col: str = "profit",
    tail_q: float = 0.9,
) -> float:
    """
    Trend μ：專門給趨勢跟蹤用的測度。

    直覺：
    - 獎勵：平均報酬、正向尾巴（大贏家）、連勝長度、樣本數
    - 溫和懲罰：過大的最大回撤 DD

    μ = mean_p * tail_gain * log(1+L) * log(1+N) * 1/(1+sqrt(DD))

    若 mean_p <= 0，直接回傳 0，表示這個 state 不值下注。
    """
    s = pd.to_numeric(df_sub[profit_col], errors="coerce").dropna()
    if s.empty:
        return 0.0

    p = s.to_numpy().astype(float)
    N = p.size
    mean_p = p.mean()
    if mean_p <= 0:
        return 0.0

    # 正向尾巴：top q 分位數以上的平均（大趨勢貢獻）
    q = np.quantile(p, tail_q)
    tail = p[p >= q]
    tail_gain = tail.mean() if tail.size > 0 else mean_p

    # 最長連勝（趨勢持續性）
    L = longest_win_run(p)

    # 最大回撤 DD（對極端回撤做 sqrt 級懲罰）
    cum = np.cumsum(p)
    peak = np.maximum.accumulate(cum)
    dd = float((peak - cum).max()) if N > 1 else 0.0
    dd_penalty = 1.0 / (1.0 + np.sqrt(max(dd, 0.0)))

    trend_mu = (
        mean_p * tail_gain * np.log(1.0 + max(L, 1)) * np.log(1.0 + N) * dd_penalty
    )
    return float(trend_mu)


# ---------- 主模組 ----------


@dataclass
class HMMTrendSelector:
    df: pd.DataFrame
    state_col: str = "HMM_State"
    profit_col: str = "profit"
    min_samples: int = 50  # 過小樣本直接視為不可信
    tail_q: float = 0.9  # 正向尾巴分位數

    def _clean_profit(self, df: Optional[pd.DataFrame] = None) -> pd.Series:
        if df is None:
            df = self.df
        return pd.to_numeric(df[self.profit_col], errors="coerce").dropna()

    # ---- 單一 state 評分 ----

    def score_states(self) -> pd.DataFrame:
        """
        回傳每個 state 的：
        - N: 樣本數
        - mean_profit
        - trend_mu
        """
        records = []
        for st in sorted(self.df[self.state_col].unique()):
            sub = self.df[self.df[self.state_col] == st]
            profits = self._clean_profit(sub)
            N = profits.size
            mean_p = profits.mean() if N > 0 else 0.0
            trend_mu = (
                compute_trend_mu(sub, profit_col=self.profit_col, tail_q=self.tail_q)
                if N >= self.min_samples
                else 0.0
            )
            records.append(
                dict(
                    state=st,
                    N=N,
                    mean_profit=mean_p,
                    trend_mu=trend_mu,
                )
            )

        out = pd.DataFrame(records).sort_values("trend_mu", ascending=False)
        out.reset_index(drop=True, inplace=True)
        return out

    # ---- 複合 state 評分 ----

    def score_combos(
        self,
        max_size: Optional[int] = None,
        include_states: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        對所有 state 組合計算 Trend μ。

        max_size: 限制組合最大長度，如 2 或 3；None 表示 1..全部。
        include_states: 若只想在特定 state 集合中做組合，可指定 [1,2,3] 之類。
        """
        if include_states is None:
            states = sorted(self.df[self.state_col].unique())
        else:
            states = sorted(include_states)

        if max_size is None:
            max_size = len(states)

        records = []
        for r in range(1, max_size + 1):
            for combo in itertools.combinations(states, r):
                sub = self.df[self.df[self.state_col].isin(combo)]
                profits = self._clean_profit(sub)
                N = profits.size
                mean_p = profits.mean() if N > 0 else 0.0
                trend_mu = (
                    compute_trend_mu(
                        sub, profit_col=self.profit_col, tail_q=self.tail_q
                    )
                    if N >= self.min_samples
                    else 0.0
                )
                records.append(
                    dict(
                        combo=combo,
                        k=len(combo),
                        N=N,
                        mean_profit=mean_p,
                        trend_mu=trend_mu,
                    )
                )

        out = pd.DataFrame(records).sort_values("trend_mu", ascending=False)
        out.reset_index(drop=True, inplace=True)
        return out

    # ---- 直接給建議 ----

    def best_states(self, top_n: int = 1) -> pd.DataFrame:
        """
        回傳趨勢 μ 最高的前 top_n 個單一 state。
        """
        scores = self.score_states()
        return scores.head(top_n)

    def best_combos(
        self, max_size: Optional[int] = None, top_n: int = 5
    ) -> pd.DataFrame:
        """
        回傳趨勢 μ 最高的前 top_n 組合。
        """
        combos = self.score_combos(max_size=max_size)
        return combos.head(top_n)
