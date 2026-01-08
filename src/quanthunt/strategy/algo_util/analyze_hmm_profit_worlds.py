import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# Config
# =====================
DATA_DIR = "/Users/zen/code/bayes_excersice/reports"
OUT_DIR = "/Users/zen/code/bayes_excersice/reports/debug_output"
os.makedirs(OUT_DIR, exist_ok=True)

PROFIT_COL = "profit"
BUY_COL = "BuySignal"
HMM_COL = "HMM_Signal"
SYMBOL_COL = "symbol"
DATE_COL = "Date"  # 可改


# =====================
# Utils
# =====================
def load_all_csv(folder):
    dfs = []
    for f in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(f)
        if SYMBOL_COL not in df.columns:
            df[SYMBOL_COL] = os.path.basename(f).split(".")[0]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def profit_stats(x):
    x = x.dropna()
    if len(x) == 0:
        return {}

    right = x[x > 0.1]
    left = x[x < 0]

    if len(right) == 0:
        cover = 0.0
    elif len(left) == 0:
        cover = np.inf
    else:
        cover = right.sum() / abs(left.sum())

    return {
        "samples": len(x),
        "median": np.median(x),
        "p95": np.percentile(x, 95),
        "p99": np.percentile(x, 99),
        "tail_cover": cover,
        "loss_rate": (x <= 0).mean(),
        "big_gain_rate": (x > 0.1).mean(),
        "right_count": len(right),
        "left_count": len(left),
    }


def layer_report(df, name):
    stats = profit_stats(df[PROFIT_COL])
    stats["layer"] = name
    return stats


def plot_profit_dist(df, title, fname):
    x = np.log(df[PROFIT_COL].dropna())
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=100, density=True)
    plt.title(title)
    plt.xlabel("log(profit)")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# =====================
# Load
# =====================
df = load_all_csv(DATA_DIR)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[PROFIT_COL])

if DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["year"] = df[DATE_COL].dt.year

print(f"Loaded rows: {len(df)}")
print(f"Symbols: {df[SYMBOL_COL].nunique()}")

# =====================
# Define Layers
# =====================
layers = {
    "L0_SystemNull": (df[HMM_COL] == 0) & (df[BUY_COL] == 0),
    "L1_HMM0_Buy1": (df[HMM_COL] == 0) & (df[BUY_COL] == 1),
    "L2_HMM1_All": (df[HMM_COL] == 1),
    "L3_Buy1_All": (df[BUY_COL] == 1),
    "L4_HMM1_Buy1": (df[HMM_COL] == 1) & (df[BUY_COL] == 1),
}

summary = []

# =====================
# Layer Reports
# =====================
for lname, cond in layers.items():
    sub = df.loc[cond].copy()
    print(f"\n[{lname}] rows = {len(sub)}")

    rep = layer_report(sub, lname)
    summary.append(rep)

    # Plot
    if len(sub) > 100:
        plot_profit_dist(
            sub,
            title=f"{lname} log-profit dist",
            fname=os.path.join(OUT_DIR, f"{lname}_dist.png"),
        )

    # Compression debug
    print("Compression ratio:", len(sub) / len(df))

    # Symbol contribution
    sym = (
        sub.groupby(SYMBOL_COL)[PROFIT_COL]
        .agg(["count", "median"])
        .sort_values("count", ascending=False)
        .head(5)
    )
    print("Top symbols:\n", sym)

# =====================
# Summary Table
# =====================
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT_DIR, "layer_summary.csv"), index=False)
print("\n=== Layer Summary ===")
print(summary_df)

# =====================
# Symbol-level Stability (Layer 4)
# =====================
print("\n=== Symbol-level check (Layer 4) ===")
l4 = df.loc[layers["L4_HMM1_Buy1"]]

sym_stats = []
for sym, g in l4.groupby(SYMBOL_COL):
    s = profit_stats(g[PROFIT_COL])
    s["symbol"] = sym
    sym_stats.append(s)

sym_df = pd.DataFrame(sym_stats).sort_values("tail_cover", ascending=False)
sym_df.to_csv(os.path.join(OUT_DIR, "symbol_L4_stats.csv"), index=False)

print(sym_df.head())

# =====================
# Temporal Stability (Layer 4)
# =====================
if "year" in df.columns:
    print("\n=== Temporal stability (Layer 4) ===")
    yearly = l4.groupby("year")[PROFIT_COL].apply(
        lambda x: profit_stats(x)["tail_cover"]
    )
    print(yearly)

    yearly.to_csv(os.path.join(OUT_DIR, "L4_tail_by_year.csv"))

# =====================
# Monte Carlo (iid, Layer 4)
# =====================
print("\n=== Monte Carlo (iid, Layer 4) ===")
returns = l4[PROFIT_COL].values
if len(returns) > 50:
    N_PATH = 5000
    N_STEP = 300
    ruin = 0

    for _ in range(N_PATH):
        path = np.random.choice(returns, N_STEP, replace=True)
        equity = np.cumprod(1 + path)
        if equity.min() < 0.5:
            ruin += 1

    print("Ruin prob (<50%):", ruin / N_PATH)

else:
    print("Not enough samples for Monte Carlo:", len(returns))


# =====================
# Open Questions Auto Dump
# =====================
with open(os.path.join(OUT_DIR, "open_questions.txt"), "w") as f:
    f.write("Auto-generated questions:\n")
    f.write("- Is Layer 4 sample size sufficient for LLN?\n")
    f.write("- Are profits dominated by few symbols?\n")
    f.write("- Monte Carlo assumes iid; loss clustering untested.\n")

print("\nDONE. Outputs in ./debug_output/")
