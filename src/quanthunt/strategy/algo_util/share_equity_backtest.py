import os, glob
import numpy as np
import pandas as pd
from quanthunt.config.core_config import config

from quanthunt.hunterverse.interface import ZERO

# =====================
# Config (改你的路徑/欄位)
# =====================
DATA_DIR = "/Users/zen/Documents/code/bayes/reports"
OUT_DIR = "/Users/zen/Documents/code/bayes/reports/fundpool_lease_replay"
os.makedirs(OUT_DIR, exist_ok=True)

DATE_COL = "Date"
MATURED_COL = "Matured"
SYMBOL_COL = "symbol"
PROFIT_COL = "profit"
BUY_COL = "BuySignal"
HMM_COL = "HMM_Signal"

START = "2023-01-01"
END = None  # None = 用資料最大時間

# 每筆固定佔用資金
STAKE = 100.0

# 初始總資金（可調）
INIT_CASH = 5000.0

# 每個 entry time 最多嘗試採納幾筆（可調）
K_PER_TICK = 3

# 只挑 L4 候選
CAND_COND = lambda df: (df[BUY_COL] == 1) & (df[HMM_COL] == 1)

# 規則選擇用的特徵（越小越好，缺就降級 random）
FEATURE_PREF = ["m_regime_noise_level", "ATR"]


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
# Prepare pool
# =====================
def build_pool(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[MATURED_COL] = pd.to_datetime(df[MATURED_COL], errors="coerce")

    need = [DATE_COL, MATURED_COL, SYMBOL_COL, PROFIT_COL, BUY_COL, HMM_COL]
    df = df.dropna(subset=need)

    # 篩時間
    df = df.sort_values(DATE_COL)
    df = df[df[DATE_COL] >= pd.Timestamp(START)]
    if END is not None:
        df = df[df[DATE_COL] <= pd.Timestamp(END)]

    # 只取候選池
    pool = df.loc[CAND_COND(df)].copy()

    # 避免同一 entry 時刻同一 symbol 重複
    pool = pool.sort_values([DATE_COL]).drop_duplicates(
        [DATE_COL, SYMBOL_COL], keep="last"
    )

    # 基本 sanity：exit 必須 >= entry（若有少數異常可選擇剔除）
    pool = pool[pool[MATURED_COL] >= pool[DATE_COL]]

    return pool


# =====================
# Selection
# =====================
def pick_random(g: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(g) <= k:
        return g
    idx = rng.choice(g.index.values, size=k, replace=False)
    return g.loc[idx]


# def pick_rule(g: pd.DataFrame, k: int, feature_col: str) -> pd.DataFrame:
#     sub = g.dropna(subset=[feature_col]).copy()
#     if sub.empty:
#         return g.head(0)
#     sub = sub.sort_values(feature_col, ascending=True)
#     return sub.head(k)


# def pick_rule(
#     g: pd.DataFrame,
#     feature_col: str,
#     q: float = 0.3,  # 取前 30%
#     min_n: int = 5,  # 樣本太少時的保護
# ) -> pd.DataFrame:
#     sub = g.dropna(subset=[feature_col]).copy()
#     n = len(sub)

#     if n == 0:
#         return g.head(0)

#     # 樣本太少 → fallback：只取 noise 最小的 1 筆
#     if n < min_n:
#         return sub.sort_values(feature_col, ascending=True).head(1)

#     # 同日橫截面 percentile
#     thresh = np.percentile(sub[feature_col].values, q * 100)

#     picked = sub[sub[feature_col] <= thresh].copy()

#     # safety：至少保留 1 筆
#     if picked.empty:
#         picked = sub.sort_values(feature_col, ascending=True).head(1)


#     return picked.sort_values(feature_col, ascending=True)
def relative_hit_rate(picked, dropped):
    if dropped.empty:
        return np.nan

    benchmark = dropped["profit"].median()
    return (picked["profit"] > benchmark).mean()


def pairwise_hit_rate(picked, dropped):
    if picked.empty or dropped.empty:
        return np.nan

    return (
        picked["profit"].values.reshape(-1, 1) > dropped["profit"].values.reshape(1, -1)
    ).mean()


def pick_rule(
    g: pd.DataFrame,
    feature_col: str,
    q: float = 0.3,  # 取前 30%
    min_n: int = 5,  # 樣本太少時的保護
    debug: bool = True,  # ★ 新增：是否輸出 debug log
    log_fn=print,  # ★ 新增：可注入 logger
) -> pd.DataFrame:

    sub = g.dropna(subset=[feature_col]).copy()
    n = len(sub)

    if n == 0:
        return g.head(0)

    # ======================
    # case 1：樣本太少
    # ======================
    if n < min_n:
        picked = sub.sort_values(feature_col, ascending=True).head(1)

        # if debug:
        #     # log_fn(f"[pick_rule] n < min_n ({n} < {min_n}), fallback to best-1")

        #     dropped = sub.drop(picked.index)

        #     picked_tbl = picked.assign(decision="PICK")[
        #         ["decision", "symbol", "Date", "profit", "m_regime_noise_level"]
        #     ]

        #     dropped_tbl = dropped.assign(decision="DROP")[
        #         ["decision", "symbol", "Date", "profit", "m_regime_noise_level"]
        #     ]

        #     log_tbl = pd.concat([picked_tbl, dropped_tbl], axis=0)

        #     if not dropped_tbl.empty:
        #         log_fn("\n . >>> . [decision table] . <<< .  ")
        #         log_fn(log_tbl.to_string(index=False))

        return picked

    # ======================
    # case 2：正常 percentile pick
    # ======================
    values = sub[feature_col].values
    thresh = np.percentile(values, q * 100)

    picked = sub[sub[feature_col] <= thresh].copy()

    # safety：至少保留 1 筆
    if picked.empty:
        picked = sub.sort_values(feature_col, ascending=True).head(1)

    if debug:
        # log_fn(f"[pick_rule] n < min_n ({n} < {min_n}), fallback to best-1")

        dropped = sub.drop(picked.index)

        picked_tbl = picked.assign(decision="PICK")[
            ["decision", "symbol", "Date", "profit", "m_regime_noise_level"]
        ]

        dropped_tbl = dropped.assign(decision="DROP")[
            ["decision", "symbol", "Date", "profit", "m_regime_noise_level"]
        ]

        log_tbl = pd.concat([picked_tbl, dropped_tbl], axis=0)

        if not dropped_tbl.empty:
            log_fn("\n . >>> . [decision table] . <<< .  ")
            log_fn(log_tbl.to_string(index=False))

            hr_pair = pairwise_hit_rate(picked, dropped)

            log_fn(
                f" . >>> . [group summary] "
                f"n_pick={len(picked)}, "
                f"n_drop={len(dropped)}, "
                f"pairwise_hit={hr_pair:.3f}"
            )
    return picked.sort_values(feature_col, ascending=True)


# =====================
# Lease-based Fund Pool Replay
# =====================
def replay_lease(pool: pd.DataFrame, selector: str, k_per_tick: int, seed: int = 7):
    rng = np.random.default_rng(seed)

    feature_col = None
    if selector == "rule":
        for c in FEATURE_PREF:
            if c in pool.columns:
                feature_col = c
                break

    # 事件時間軸：把所有 entry 時刻排序
    times = pool[DATE_COL].sort_values().unique()

    available_cash = float(INIT_CASH)
    # active leases: list of dict(end, notional, profit, symbol, entry)
    active = []

    records = []
    profits = []
    hit_stats = []

    # 為了快速釋放：每個時間點要釋放哪些 lease（用 dict bucket）
    # 注意：同一個成熟時間可能有多筆 lease
    # 這裡簡化：每次 tick 掃 active 一次（樣本大時可改 bucket）
    for now in times:
        now = pd.Timestamp(now)

        # Phase 1: release matured leases (end <= now)
        if active:
            still_active = []
            released = 0
            released_cash = 0.0
            for lease in active:
                if lease["end"] <= now:
                    payout = lease["notional"] * (1.0 + lease["profit"])
                    available_cash += payout
                    released += 1
                    released_cash += payout
                    profits.append(lease["profit"])
                    # print(
                    #     f'[{lease["entry"]} - {lease["end"]}]: [{lease["symbol"]}] {payout:.2f} ({lease["profit"]:.3f})'
                    # )
                else:
                    still_active.append(lease)
            active = still_active
        else:
            released = 0
            released_cash = 0.0

        # Phase 2: select candidates at this entry time
        g = pool.loc[pool[DATE_COL] == now].copy()
        n_candidates = len(g)

        if n_candidates == 0:
            equity = available_cash + sum(x["notional"] for x in active)
            records.append(
                {
                    "time": now,
                    "equity": equity,
                    "available_cash": available_cash,
                    "active_leases": len(active),
                    "released": released,
                    "released_cash": released_cash,
                    "n_candidates": 0,
                    "n_picked": 0,
                    "n_accepted": 0,
                    "n_rejected_cash": 0,
                    "n_coverage": 0,
                }
            )
            continue

        if selector == "random":
            picked = pick_random(g, k_per_tick, rng)
        elif selector == "rule":
            if feature_col is None:
                picked = pick_random(g, k_per_tick, rng)
            else:
                picked = pick_rule(g, feature_col, q=0.5, min_n=k_per_tick)
                regime_val = g["m_regime_noise_level"].median()

                # 驗證 pick 是否有效
                dropped = g.drop(picked.index, errors="ignore")
                hr_pair = pairwise_hit_rate(picked, dropped)
                hit_stats.append(
                    {
                        "Date": now,
                        "feature": feature_col,
                        "pairwise_hit": hr_pair,
                        "n_pick": len(picked),
                        "n_drop": len(dropped),
                        "n_candidates": len(g),
                        "m_regime_noise_level": regime_val,  # ★ 加這個
                    }
                )
                # 若因缺值選不滿，隨機補齊
                need = min(k_per_tick, len(g)) - len(picked)
                if need > 0:
                    remain = g.drop(index=picked.index, errors="ignore")
                    if len(remain) > 0:
                        picked2 = pick_random(remain, min(need, len(remain)), rng)
                        picked = pd.concat([picked, picked2], ignore_index=False)
        else:
            raise ValueError("selector must be 'random' or 'rule'")

        n_picked = len(picked)
        n_accepted = 0
        n_rejected_cash = 0

        # 依序嘗試採納（這裡就像 fund pool 批次裁決後逐筆 grant）
        # 規則：每筆固定 STAKE，錢不夠就拒絕
        for _, row in picked.sort_values(DATE_COL).iterrows():
            if available_cash >= STAKE:
                available_cash -= STAKE
                active.append(
                    {
                        "entry": now,
                        "end": pd.Timestamp(row[MATURED_COL]),
                        "notional": float(STAKE),
                        "profit": float(row[PROFIT_COL]),
                        "symbol": row[SYMBOL_COL],
                    }
                )
                n_accepted += 1
            else:
                n_rejected_cash += 1

        # Phase 3: record equity (帳面=可用+鎖住本金；profit 等到 end 才進來)
        equity = available_cash + sum(x["notional"] for x in active)
        records.append(
            {
                "time": now,
                "equity": equity,
                "available_cash": available_cash,
                "active_leases": len(active),
                "released": released,
                "released_cash": released_cash,
                "n_candidates": n_candidates,
                "n_picked": n_picked,
                "n_accepted": n_accepted,
                "n_rejected_cash": n_rejected_cash / (n_candidates + ZERO),
                "n_coverage": n_accepted / (n_candidates + ZERO),
            }
        )

    # for lease in active:
    #     print(
    #         f'[{lease["entry"]} - {lease["end"]}]: [{lease["symbol"]}] {payout:.2f} ({lease["profit"]:.3f})'
    #     )

    curve = pd.DataFrame(records).sort_values("time")

    # Max drawdown on equity (帳面值)
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    curve["median_profit"] = np.median(profits)
    return curve, feature_col, pd.DataFrame(hit_stats)


def summarize(curve: pd.DataFrame) -> dict:
    if curve.empty:
        return {}
    return {
        "final_equity": float(curve["equity"].iloc[-1]),
        "max_dd": float(curve["drawdown"].min()),
        "min_cash": float(curve["available_cash"].min()),
        "avg_active_leases": float(curve["active_leases"].mean()),
        "avg_candidates": float(curve["n_candidates"].mean()),
        "avg_accepted": float(curve["n_accepted"].mean()),
        "cash_reject_rate": float((curve["n_rejected_cash"] > 0).mean()),
        "avg_coverage": float((curve["n_coverage"] > 0).mean()),
        "daily_opportunity": float((curve["n_picked"] > 0).mean()),
        "median_profit": float((curve["median_profit"]).mean()),
        "ticks": int(len(curve)),
    }


if __name__ == "__main__":
    df = load_all_csv(DATA_DIR)
    pool = build_pool(df)

    print(f"Pool rows (L4 candidates): {len(pool)}")
    print(f"Symbols in pool: {pool[SYMBOL_COL].nunique()}")
    print(f"Time ticks in pool: {pool[DATE_COL].nunique()}")
    print("Holding time (hours) summary:")
    hold_hours = (pool[MATURED_COL] - pool[DATE_COL]).dt.total_seconds() / 3600.0
    print(hold_hours.describe())

    # Replay: random
    curve_rand, feat_rand, hit_df = replay_lease(
        pool, selector="random", k_per_tick=K_PER_TICK, seed=7
    )
    curve_rand.to_csv(os.path.join(OUT_DIR, "equity_lease_random.csv"), index=False)

    # Replay: rule
    curve_rule, feat_rule, hit_df = replay_lease(
        pool, selector="rule", k_per_tick=K_PER_TICK, seed=7
    )
    hit_df.to_csv(os.path.join(OUT_DIR, "hit_stats.csv"), index=False)
    curve_rule.to_csv(os.path.join(OUT_DIR, "equity_lease_rule.csv"), index=False)

    s_rand = summarize(curve_rand)
    s_rule = summarize(curve_rule)

    summary = pd.DataFrame(
        [
            {"selector": "random", "rule_feature": feat_rand, **s_rand},
            {"selector": "rule", "rule_feature": feat_rule, **s_rule},
        ]
    )
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    print("\n=== Summary ===")
    print(summary)
    hit_df["rolling_20"] = hit_df["pairwise_hit"].rolling(20, min_periods=10).mean()
    hit_df["noise_bucket"] = pd.qcut(
        hit_df["m_regime_noise_level"],
        q=4,
        labels=["low", "mid-low", "mid-high", "high"],
    )
    print(hit_df[-60:])
    print(hit_df.groupby("noise_bucket")["pairwise_hit"].mean())

    print(f"\nDONE. Outputs in: {OUT_DIR}")
