import os, glob
import numpy as np
import pandas as pd

# =====================
# Config (請改成你的)
# =====================
DATA_DIR = "/Users/zen/code/bayes_excersice/reports"
OUT_DIR = "/Users/zen/code/bayes_excersice/reports/fundpool_replay"
os.makedirs(OUT_DIR, exist_ok=True)

# 必要欄位（依你目前檔案）
DATE_COL = "Date"
SYMBOL_COL = "symbol"
PROFIT_COL = "profit"
BUY_COL = "BuySignal"
HMM_COL = "HMM_Signal"

# 候選條件：你目前最在意的 L4
CAND_COND = lambda df: (df[BUY_COL] == 1) & (df[HMM_COL] == 1)

# 每天最多採納幾筆
K_PER_DAY = 3

# 回測區間（可調）
START = "2015-01-01"
END = None  # None = 用資料最大值

# 規則用特徵（有就用，沒有就降級）
FEATURE_PREF = ["m_regime_noise_level", "ATR"]  # 越小越好（先做 baseline）


# =====================
# IO
# =====================
def load_all_csv(folder):
    dfs = []
    for f in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(f)
        if SYMBOL_COL not in df.columns:
            df[SYMBOL_COL] = os.path.basename(f).split(".")[0]
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No CSV found in {folder}")
    return pd.concat(dfs, ignore_index=True)


# =====================
# Daily Candidate Pool
# =====================
def build_candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).copy()

    # 日期
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, SYMBOL_COL, PROFIT_COL, BUY_COL, HMM_COL])

    # 篩區間
    df = df.sort_values(DATE_COL)
    df = df[df[DATE_COL] >= pd.Timestamp(START)]
    if END is not None:
        df = df[df[DATE_COL] <= pd.Timestamp(END)]

    # 候選池：符合條件的 row
    pool = df.loc[CAND_COND(df)].copy()

    # 同一天同一 symbol 若有重複 row（例如資料重複），保留最後一筆
    pool = pool.sort_values([DATE_COL]).drop_duplicates(
        [DATE_COL, SYMBOL_COL], keep="last"
    )

    # 加上 day key
    pool["day"] = pool[DATE_COL].dt.floor("D")

    return pool


# =====================
# Selection Rules
# =====================
def pick_random(day_df: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(day_df) <= k:
        return day_df
    idx = rng.choice(day_df.index.values, size=k, replace=False)
    return day_df.loc[idx]


def pick_rule(day_df: pd.DataFrame, k: int, feature_col: str) -> pd.DataFrame:
    # 越小越好（先用 noise / ATR baseline）
    # 若 feature 缺失太多，降級成隨機的效果（在外層處理）
    sub = day_df.dropna(subset=[feature_col]).copy()
    if sub.empty:
        return day_df.head(0)  # 外層會 fallback
    sub = sub.sort_values(feature_col, ascending=True)
    return sub.head(k)


# =====================
# Fund Pool Replay
# =====================
def replay(
    pool: pd.DataFrame, selector: str, k_per_day: int, seed: int = 7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 決定規則要用哪個 feature
    feature_col = None
    if selector == "rule":
        for c in FEATURE_PREF:
            if c in pool.columns:
                feature_col = c
                break

    equity = 1.0
    rows = []

    for day, g in pool.groupby("day", sort=True):
        g = g.copy()

        picked = None
        if selector == "random":
            picked = pick_random(g, k_per_day, rng)
        elif selector == "rule":
            if feature_col is None:
                # 沒 feature 就退化成隨機
                picked = pick_random(g, k_per_day, rng)
            else:
                picked = pick_rule(g, k_per_day, feature_col)
                # 若因缺值導致沒選到，退化成隨機補滿
                if len(picked) < min(k_per_day, len(g)):
                    remain = g.drop(index=picked.index, errors="ignore")
                    need = min(k_per_day - len(picked), len(remain))
                    if need > 0:
                        picked2 = pick_random(remain, need, rng)
                        picked = pd.concat([picked, picked2], ignore_index=False)
        else:
            raise ValueError("selector must be 'random' or 'rule'")

        if picked is None or picked.empty:
            # 當天不交易
            rows.append(
                {
                    "day": day,
                    "equity": equity,
                    "n_candidates": len(g),
                    "n_picked": 0,
                    "day_ret": 0.0,
                }
            )
            continue

        # 等權投入：當天報酬 = picked profit 平均
        day_ret = float(np.nanmean(picked[PROFIT_COL].values))
        equity *= 1.0 + day_ret

        rows.append(
            {
                "day": day,
                "equity": equity,
                "n_candidates": int(len(g)),
                "n_picked": int(len(picked)),
                "day_ret": day_ret,
            }
        )

    out = pd.DataFrame(rows).sort_values("day")
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out


def summarize_curve(curve: pd.DataFrame) -> dict:
    if curve.empty:
        return {"final_equity": 1.0, "max_dd": 0.0, "mean_day_ret": 0.0, "days": 0}
    return {
        "final_equity": float(curve["equity"].iloc[-1]),
        "max_dd": float(curve["drawdown"].min()),
        "mean_day_ret": float(curve["day_ret"].mean()),
        "days": int(len(curve)),
        "avg_candidates": float(curve["n_candidates"].mean()),
        "avg_picked": float(curve["n_picked"].mean()),
    }


if __name__ == "__main__":
    df = load_all_csv(DATA_DIR)
    pool = build_candidate_pool(df)

    print(f"Pool rows (L4 candidates): {len(pool)}")
    print(f"Symbols in pool: {pool[SYMBOL_COL].nunique()}")
    print(f"Days in pool: {pool['day'].nunique()}")

    # Replay: Random baseline
    curve_rand = replay(pool, selector="random", k_per_day=K_PER_DAY, seed=7)
    s_rand = summarize_curve(curve_rand)
    curve_rand.to_csv(os.path.join(OUT_DIR, "equity_random.csv"), index=False)

    # Replay: Rule-based
    curve_rule = replay(pool, selector="rule", k_per_day=K_PER_DAY, seed=7)
    s_rule = summarize_curve(curve_rule)
    curve_rule.to_csv(os.path.join(OUT_DIR, "equity_rule.csv"), index=False)

    # Summary
    summary = pd.DataFrame(
        [
            {"selector": "random", **s_rand},
            {"selector": "rule", **s_rule},
        ]
    )
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    print("\n=== Summary ===")
    print(summary)

    # 額外：把每天候選數分佈輸出
    daily_candidates = (
        pool.groupby("day")[SYMBOL_COL].count().rename("n_candidates").reset_index()
    )
    daily_candidates.to_csv(os.path.join(OUT_DIR, "daily_candidates.csv"), index=False)

    print(f"\nDONE. Outputs in: {OUT_DIR}")
