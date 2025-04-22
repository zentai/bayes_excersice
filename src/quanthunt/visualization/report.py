import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click


def visualize_backtest(df, window_size=60):
    # Ensure Date and Matured columns are in datetime format
    df["Date"] = pd.to_datetime(df["Date"])
    df["Matured"] = pd.to_datetime(df["Matured"])

    # Set up the visualization style
    sns.set(style="whitegrid")

    # Capture Rate Calculation
    high_profit_trades = (df["P/L"] > 3).sum()
    captured_high_profit_trades = ((df["BuySignal"] == True) & (df["P/L"] > 3)).sum()
    capture_rate = (
        captured_high_profit_trades / high_profit_trades
        if high_profit_trades > 0
        else 0
    )

    # Signal Coverage Rate Calculation
    total_trading_days = len(df)
    signal_days = df["BuySignal"].fillna(False).astype(int).sum()
    signal_coverage_rate = (
        signal_days / total_trading_days if total_trading_days > 0 else 0
    )

    # Average Holding Period Calculation
    df["holding_period"] = (
        df["Matured"] - df["Date"]
    ).dt.total_seconds() / 3600  # Convert to hours
    average_holding_period = (
        df["holding_period"].mean() if not df["holding_period"].isna().all() else 0
    )

    # Max Consecutive Wins/Losses Calculation using pandas
    df["win"] = df["P/L"] > 0
    df["loss"] = df["P/L"] <= 0
    # Calculate consecutive wins and losses
    df["consecutive_wins"] = (
        df["win"]
        .astype(int)
        .groupby((df["win"] != df["win"].shift()).cumsum())
        .cumsum()
    )
    df["consecutive_losses"] = (
        df["loss"]
        .astype(int)
        .groupby((df["loss"] != df["loss"].shift()).cumsum())
        .cumsum()
    )
    # Get maximum consecutive wins and losses
    max_consecutive_wins = df["consecutive_wins"].max()
    max_consecutive_losses = df["consecutive_losses"].max()

    # Win Rate Trend Calculation using moving window
    df["win_numeric"] = df["win"].astype(int)
    win_rate_trend = df["win_numeric"].rolling(window=window_size, min_periods=1).mean()

    # Expected Value and Breakeven Point Calculation
    win_rate = df["win"].mean() if not df["win"].isna().all() else 0
    fail_rate = 1 - win_rate

    avg_profit = df.loc[df["P/L"] > 0, "P/L"].mean() if (df["P/L"] > 0).any() else 0
    avg_loss = -df.loc[df["P/L"] <= 0, "P/L"].mean() if (df["P/L"] <= 0).any() else 0

    expected_value = (win_rate * avg_profit) - (fail_rate * avg_loss)

    profit_loss_ratio = avg_profit / avg_loss if avg_loss != 0 else 0
    breakeven_win_rate = 1 / (1 + profit_loss_ratio) if profit_loss_ratio != 0 else 0

    # Print summary metrics
    print("Summary Metrics:")
    print(f"Capture Rate: {capture_rate:.3f}")
    print(f"Signal Coverage Rate: {signal_coverage_rate:.3f}")
    print(f"Average Holding Period (hours): {average_holding_period:.2f}")
    print(f"Max Consecutive Wins: {max_consecutive_wins}")
    print(f"Max Consecutive Losses: {max_consecutive_losses}")
    print(f"Expected Value per Trade: {expected_value:.3f}")
    print(f"Breakeven Win Rate: {breakeven_win_rate:.3f}")

    # P/L Binning into 10 segments
    pl_min = 0
    pl_max = df["P/L"].max()
    bins = np.linspace(
        pl_min, pl_max, 11
    )  # Split into 10 parts (11 points create 10 bins)
    print(bins)
    df["P/L_bins"] = pd.cut(df["P/L"], bins=bins, include_lowest=True)

    # Count occurrences for each bin using P/L_bins
    captured_counts = df.loc[df["BuySignal"] == True].groupby("P/L_bins").size()
    total_counts = df.groupby("P/L_bins").size()

    # Combine into a summary DataFrame
    summary_df = pd.DataFrame(
        {
            "P/L_bin": captured_counts.index.astype(str),
            "Captured_Counts": captured_counts,
            "Total_Counts": total_counts,
        }
    ).fillna(0)

    # Print the summary DataFrame
    print("Summary of Captured vs Total Counts per P/L Bin:")
    print(summary_df)

    # Profit and Loss Distribution Visualization (side-by-side bar chart)
    x = np.arange(len(captured_counts.index))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2, captured_counts, width, label="Captured by BuySignal", alpha=0.6
    )
    bars2 = ax.bar(
        x + width / 2, total_counts, width, label="Total High P/L", alpha=0.3
    )

    # Add some text for labels, title and axes ticks
    ax.set_xlabel("P/L Bins")
    ax.set_ylabel("Number of Trades")
    ax.set_title("P/L Distribution for Captured vs. Total High Profit Trades")
    ax.set_xticks(x)
    ax.set_xticklabels(captured_counts.index.astype(str), rotation=45)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

    # Calculating Capture Rate by bin
    capture_rate_by_bin = (captured_counts / total_counts).fillna(0)

    # Capture Rate Trend Over P/L Bins Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(
        capture_rate_by_bin.index.astype(str),
        capture_rate_by_bin,
        marker="o",
        linestyle="-",
        color="g",
    )
    plt.title("Capture Rate Trend Over P/L Bins")
    plt.xlabel("P/L Bins")
    plt.ylabel("Capture Rate")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Win Rate Trend Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], win_rate_trend, marker="o", linestyle="-", color="b")
    plt.title("Win Rate Trend Over Time (Moving Window)")
    plt.xlabel("Date")
    plt.ylabel("Win Rate")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


@click.command()
@click.argument("filename")
@click.option("--window", default=60, help="Window size for win rate smoothing")
def main(filename, window):
    df = pd.read_csv(filename)
    visualize_backtest(df, window)


if __name__ == "__main__":
    main()
