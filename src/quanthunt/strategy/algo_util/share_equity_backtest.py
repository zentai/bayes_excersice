import os, glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

from quanthunt.hunterverse.interface import ZERO


# ============================================================
# Performance Panel Spec v1.0 (Minimal Refactor)
# - Keep logic unchanged
# - Split outputs into 5 panels (CSV) + optional legacy CSVs
#
# Panels:
# 0) run_meta.csv
# 1) outcome_summary.csv
# 2) execution_timeseries.csv
# 3) selection_quality.csv
# 4) risk_structure.csv
# ============================================================


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
CAND_COND: Callable[[pd.DataFrame], pd.Series] = lambda df: (df[BUY_COL] == 1) & (
    df[HMM_COL] == 1
)

# 規則選擇用的特徵（越小越好，缺就降級 random）
FEATURE_PREF = ["m_regime_noise_level", "ATR"]

# 寫 legacy 輸出（保留你原本檔名，方便比對）；不需要可關掉
WRITE_LEGACY = True

# pick_rule debug（會很吵，預設關）
DEBUG_PICK_RULE = False
DEBUG_LOG_PATH = os.path.join(OUT_DIR, "decision_debug.txt")


# =====================
# Helpers
# =====================
def _now_str() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


class FileLogger:
    """Simple line logger to a file (append)."""

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        if enabled:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(
                    f"# decision debug log\n# created_at_utc={pd.Timestamp.utcnow()}\n\n"
                )

    def __call__(self, msg: str):
        if not self.enabled:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg)
            if not msg.endswith("\n"):
                f.write("\n")


# =====================
# IO
# =====================
def load_all_csv(folder: str) -> pd.DataFrame:
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
def build_pool(df: pd.DataFrame, start: str, end: Optional[str]) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[MATURED_COL] = pd.to_datetime(df[MATURED_COL], errors="coerce")

    need = [DATE_COL, MATURED_COL, SYMBOL_COL, PROFIT_COL, BUY_COL, HMM_COL]
    df = df.dropna(subset=need)

    # 篩時間
    df = df.sort_values(DATE_COL)
    df = df[df[DATE_COL] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df[DATE_COL] <= pd.Timestamp(end)]

    # 只取候選池
    pool = df.loc[CAND_COND(df)].copy()

    # 避免同一 entry 時刻同一 symbol 重複
    pool = pool.sort_values([DATE_COL]).drop_duplicates(
        [DATE_COL, SYMBOL_COL], keep="last"
    )

    return pool


# =====================
# Selection
# =====================
def pick_random(g: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(g) <= k:
        return g
    idx = rng.choice(g.index.values, size=k, replace=False)
    return g.loc[idx]


def pairwise_hit_rate(picked: pd.DataFrame, dropped: pd.DataFrame) -> float:
    if picked.empty or dropped.empty:
        return np.nan
    return (
        picked[PROFIT_COL].values.reshape(-1, 1)
        > dropped[PROFIT_COL].values.reshape(1, -1)
    ).mean()


def pick_rule(
    g: pd.DataFrame,
    feature_col: str,
    q: float = 0.3,  # 取前 30%
    min_n: int = 5,  # 樣本太少時的保護
    debug: bool = False,
    log_fn=print,
) -> pd.DataFrame:
    sub = g.dropna(subset=[feature_col]).copy()
    n = len(sub)
    if n == 0:
        return g.head(0)

    # case 1：樣本太少 → best-1
    if n < min_n:
        picked = sub.sort_values(feature_col, ascending=True).head(1)
        return picked

    # case 2：percentile pick
    values = sub[feature_col].values
    thresh = np.percentile(values, q * 100)
    picked = sub[sub[feature_col] <= thresh].copy()

    if picked.empty:
        picked = sub.sort_values(feature_col, ascending=True).head(1)

    if debug:
        dropped = sub.drop(picked.index)
        if not dropped.empty:
            picked_tbl = picked.assign(decision="PICK")[
                ["decision", SYMBOL_COL, DATE_COL, PROFIT_COL, feature_col]
            ]
            dropped_tbl = dropped.assign(decision="DROP")[
                ["decision", SYMBOL_COL, DATE_COL, PROFIT_COL, feature_col]
            ]
            log_tbl = pd.concat([picked_tbl, dropped_tbl], axis=0)
            log_fn("\n . >>> . [decision table] . <<< .  ")
            log_fn(log_tbl.to_string(index=False))

            hr_pair = pairwise_hit_rate(picked, dropped)
            log_fn(
                f" . >>> . [group summary] n_pick={len(picked)}, n_drop={len(dropped)}, pairwise_hit={hr_pair:.3f}"
            )

    return picked.sort_values(feature_col, ascending=True)


# =====================
# Lease-based Fund Pool Replay
# =====================
@dataclass
class ReplayConfig:
    data_dir: str
    out_dir: str
    start: str
    end: Optional[str]
    init_cash: float
    stake: float
    k_per_tick: int
    feature_pref: List[str]
    selector: str
    seed: int
    debug_pick_rule: bool = False


def replay_lease(
    pool: pd.DataFrame, cfg: ReplayConfig, debug_logger: Optional[FileLogger] = None
):
    rng = np.random.default_rng(cfg.seed)

    feature_col = None
    if cfg.selector == "rule":
        for c in cfg.feature_pref:
            if c in pool.columns:
                feature_col = c
                break

    times = pool[DATE_COL].sort_values().unique()

    available_cash = float(cfg.init_cash)
    active: List[Dict] = []  # {entry, end, notional, profit, symbol}

    records = []
    profits = []
    hit_stats = []

    for now in times:
        now = pd.Timestamp(now)

        # Phase 1: release matured leases
        released = 0
        released_cash = 0.0
        if active:
            still_active = []
            for lease in active:
                if lease["end"] <= now:
                    payout = lease["notional"] * (1.0 + lease["profit"])
                    available_cash += payout
                    released += 1
                    released_cash += payout
                    profits.append(lease["profit"])
                else:
                    still_active.append(lease)
            active = still_active

        # Phase 2: candidates at this tick
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
                    "n_rejected_cash": 0.0,
                    "n_coverage": 0.0,
                }
            )
            continue

        # Phase 2.1: pick
        if cfg.selector == "random":
            picked = pick_random(g, cfg.k_per_tick, rng)
        elif cfg.selector == "rule":
            if feature_col is None:
                picked = pick_random(g, cfg.k_per_tick, rng)
            else:
                log_fn = debug_logger if debug_logger is not None else print
                picked = pick_rule(
                    g,
                    feature_col,
                    q=0.5,
                    min_n=cfg.k_per_tick,
                    debug=cfg.debug_pick_rule,
                    log_fn=log_fn,
                )

                dropped = g.drop(picked.index, errors="ignore")
                hr_pair = pairwise_hit_rate(picked, dropped)

                regime_val = (
                    g["m_regime_noise_level"].median()
                    if "m_regime_noise_level" in g.columns
                    else np.nan
                )
                hit_stats.append(
                    {
                        "Date": now,
                        "feature": feature_col,
                        "pairwise_hit": hr_pair,
                        "n_pick": len(picked),
                        "n_drop": len(dropped),
                        "n_candidates": len(g),
                        "m_regime_noise_level": regime_val,
                    }
                )

                # 若因缺值選不滿，隨機補齊
                need = min(cfg.k_per_tick, len(g)) - len(picked)
                if need > 0:
                    remain = g.drop(index=picked.index, errors="ignore")
                    if len(remain) > 0:
                        picked2 = pick_random(remain, min(need, len(remain)), rng)
                        picked = pd.concat([picked, picked2], ignore_index=False)
        else:
            raise ValueError("selector must be 'random' or 'rule'")

        n_picked = len(picked)

        # Phase 2.2: accept sequentially
        n_accepted = 0
        n_rejected_cash = 0
        for _, row in picked.sort_values(DATE_COL).iterrows():
            if available_cash >= cfg.stake:
                available_cash -= cfg.stake
                active.append(
                    {
                        "entry": now,
                        "end": pd.Timestamp(row[MATURED_COL]),
                        "notional": float(cfg.stake),
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

    curve = pd.DataFrame(records).sort_values("time")
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    curve["median_profit"] = float(np.median(profits)) if len(profits) else np.nan

    open_trades = pd.DataFrame(
        [
            {
                "symbol": x["symbol"],
                "entry_time": x["entry"],
                "matured_time": x["end"],
                "notional": x["notional"],
                "profit_if_matured": x["profit"],
                "expected_payout": x["notional"] * (1.0 + x["profit"]),
            }
            for x in active
        ]
    )

    return curve, feature_col, pd.DataFrame(hit_stats), open_trades


# =====================
# Summary (Outcome / Execution KPI)
# =====================
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
        # NOTE: 這裡沿用你原本寫法（>0 的天數比），語義是 “有 coverage 的 tick 比例”
        "avg_coverage": float((curve["n_coverage"] > 0).mean()),
        "daily_opportunity": float((curve["n_picked"] > 0).mean()),
        "median_profit": float(curve["median_profit"].iloc[-1]),
        "ticks": int(len(curve)),
    }


# =====================
# Panel Writers (Spec v1.0)
# =====================
def write_run_meta(out_dir: str, meta: Dict):
    pd.DataFrame([meta]).to_csv(os.path.join(out_dir, "run_meta.csv"), index=False)


def write_outcome_panel(out_dir: str, summary_df: pd.DataFrame):
    cols = [
        "selector",
        "rule_feature",
        "final_equity",
        "max_dd",
        "median_profit",
        "ticks",
    ]
    summary_df[cols].to_csv(os.path.join(out_dir, "outcome_summary.csv"), index=False)


def write_execution_panel(out_dir: str, curve: pd.DataFrame):
    cols = [
        "time",
        "available_cash",
        "active_leases",
        "n_candidates",
        "n_accepted",
        "n_rejected_cash",
        "n_coverage",
        "equity",
        "drawdown",
    ]
    out = curve[cols].rename(columns={"n_coverage": "coverage_ratio"})
    out.to_csv(os.path.join(out_dir, "execution_timeseries.csv"), index=False)


def write_selection_quality_panel(out_dir: str, hit_df: pd.DataFrame):
    if hit_df is None or hit_df.empty:
        return

    df = hit_df.copy()
    df["rolling_pairwise_hit"] = df["pairwise_hit"].rolling(20, min_periods=10).mean()

    keep_cols = [
        "Date",
        "feature",
        "pairwise_hit",
        "rolling_pairwise_hit",
        "n_pick",
        "n_drop",
        "n_candidates",
        "m_regime_noise_level",
    ]
    df[keep_cols].to_csv(os.path.join(out_dir, "selection_quality.csv"), index=False)


def write_risk_structure_panel(out_dir: str, curve: pd.DataFrame):
    """
    Minimal v1 risk structure:
    - concurrent positions distribution proxy
    - tail drawdown
    """
    risk = {
        "max_concurrent_positions": float(curve["active_leases"].max()),
        "avg_concurrent_positions": float(curve["active_leases"].mean()),
        "p95_drawdown": float(curve["drawdown"].quantile(0.05)),  # worst 5%
        "p99_drawdown": float(curve["drawdown"].quantile(0.01)),  # worst 1%
        "min_cash": float(curve["available_cash"].min()),
    }
    pd.DataFrame([risk]).to_csv(
        os.path.join(out_dir, "risk_structure.csv"), index=False
    )


# =====================
# Legacy Writer (optional)
# =====================
def write_legacy(
    out_dir: str,
    curve_rand: pd.DataFrame,
    curve_rule: pd.DataFrame,
    hit_rule: pd.DataFrame,
    summary: pd.DataFrame,
):
    curve_rand.to_csv(os.path.join(out_dir, "equity_lease_random.csv"), index=False)
    curve_rule.to_csv(os.path.join(out_dir, "equity_lease_rule.csv"), index=False)
    if hit_rule is not None and not hit_rule.empty:
        hit_rule.to_csv(os.path.join(out_dir, "hit_stats.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)


# =====================
# Main
# =====================
if __name__ == "__main__":
    _safe_makedirs(OUT_DIR)

    # Decision debug logger (only if DEBUG_PICK_RULE=True)
    dbg_logger = FileLogger(DEBUG_LOG_PATH, enabled=DEBUG_PICK_RULE)

    df = load_all_csv(DATA_DIR)
    pool = build_pool(df, start=START, end=END)

    print(f"Pool rows (L4 candidates): {len(pool)}")
    print(f"Symbols in pool: {pool[SYMBOL_COL].nunique()}")
    print(f"Time ticks in pool: {pool[DATE_COL].nunique()}")
    print("Holding time (hours) summary:")
    hold_hours = (pool[MATURED_COL] - pool[DATE_COL]).dt.total_seconds() / 3600.0
    print(hold_hours.describe())

    # ---------------------
    # Replays
    # ---------------------
    cfg_rand = ReplayConfig(
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        start=START,
        end=END,
        init_cash=INIT_CASH,
        stake=STAKE,
        k_per_tick=K_PER_TICK,
        feature_pref=FEATURE_PREF,
        selector="random",
        seed=7,
        debug_pick_rule=False,
    )
    curve_rand, feat_rand, hit_rand, open_trades = replay_lease(
        pool, cfg_rand, debug_logger=dbg_logger
    )

    cfg_rule = ReplayConfig(
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        start=START,
        end=END,
        init_cash=INIT_CASH,
        stake=STAKE,
        k_per_tick=K_PER_TICK,
        feature_pref=FEATURE_PREF,
        selector="rule",
        seed=7,
        debug_pick_rule=DEBUG_PICK_RULE,
    )
    curve_rule, feat_rule, hit_rule, open_trades = replay_lease(
        pool, cfg_rule, debug_logger=dbg_logger
    )

    # ---------------------
    # Summaries
    # ---------------------
    s_rand = summarize(curve_rand)
    s_rule = summarize(curve_rule)

    summary = pd.DataFrame(
        [
            {"selector": "random", "rule_feature": feat_rand, **s_rand},
            {"selector": "rule", "rule_feature": feat_rule, **s_rule},
        ]
    )

    # ---------------------
    # Panel 0: Run Meta
    # ---------------------
    run_meta = {
        "run_id": _now_str(),
        "data_dir": DATA_DIR,
        "out_dir": OUT_DIR,
        "start_date": START,
        "end_date": END,
        "INIT_CASH": INIT_CASH,
        "STAKE": STAKE,
        "K_PER_TICK": K_PER_TICK,
        "selector_rule_feature": feat_rule,
        "universe_size": int(pool[SYMBOL_COL].nunique()),
        "pool_rows": int(len(pool)),
    }
    write_run_meta(OUT_DIR, run_meta)

    # ---------------------
    # Panel 1: Outcome Summary
    # ---------------------
    write_outcome_panel(OUT_DIR, summary)

    # ---------------------
    # Panel 2: Execution (use RULE as primary)
    # ---------------------
    write_execution_panel(OUT_DIR, curve_rule)

    # ---------------------
    # Panel 3: Selection Quality (use RULE)
    # ---------------------
    write_selection_quality_panel(OUT_DIR, hit_rule)

    # ---------------------
    # Panel 4: Risk Structure (minimal, use RULE)
    # ---------------------
    write_risk_structure_panel(OUT_DIR, curve_rule)

    # ---------------------
    # Panel 5: Open positions
    # ---------------------
    if not open_trades.empty:
        as_of_time = curve_rule["time"].iloc[-1]

        open_trades = open_trades.copy()
        open_trades["as_of_time"] = as_of_time
        open_trades["holding_hours"] = (
            as_of_time - pd.to_datetime(open_trades["entry_time"])
        ).dt.total_seconds() / 3600.0
        open_trades["remaining_hours"] = (
            pd.to_datetime(open_trades["matured_time"]) - as_of_time
        ).dt.total_seconds() / 3600.0

        open_trades["age_bucket"] = pd.cut(
            open_trades["holding_hours"],
            bins=[-np.inf, 24, 72, np.inf],
            labels=["short(<1d)", "mid(1-3d)", "long(>3d)"],
        )

        open_trades.to_csv(os.path.join(OUT_DIR, "open_trades.csv"), index=False)

    # ---------------------
    # Optional Legacy Outputs
    # ---------------------
    if WRITE_LEGACY:
        write_legacy(OUT_DIR, curve_rand, curve_rule, hit_rule, summary)

    print(f"\nDONE. Panels written to: {OUT_DIR}")
    print("Panels:")
    print(" - run_meta.csv")
    print(" - outcome_summary.csv")
    print(" - execution_timeseries.csv")
    print(" - selection_quality.csv")
    print(" - risk_structure.csv")
    print(" - open_trades.csv")
    if WRITE_LEGACY:
        print("\nLegacy:")
        print(" - equity_lease_random.csv")
        print(" - equity_lease_rule.csv")
        print(" - hit_stats.csv")
        print(" - summary.csv")
    if DEBUG_PICK_RULE:
        print(f"\nDebug decision log: {DEBUG_LOG_PATH}")
