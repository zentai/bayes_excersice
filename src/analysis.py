import matplotlib.pyplot as plt
import pandas as pd
from config import config

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir


def split_buy_sell(df):
    # Create a list to hold each buy-sell period
    buy_sell_periods = []

    # Initialize variables to track buy and sell row indices
    buy_start = None

    # Iterate through the rows of the dataframe
    for i, row in df.iterrows():
        # Identify a 'buy' record (sBuy has value, and sSell and sProfit are NaN)
        if (
            not pd.isna(row["sBuy"])
            and pd.isna(row["sSell"])
            and pd.isna(row["sProfit"])
        ):
            if buy_start is None:
                buy_start = i  # Set the buy start index if not already set

        # Identify a 'sell' record (sBuy has value, and sSell and sProfit have values)
        elif not pd.isna(row["sSell"]) and not pd.isna(row["sProfit"]):
            if buy_start is not None:
                # Extract rows from buy_start to the current row (sell)
                buy_sell_period = df.loc[buy_start:i]
                # Check if the total number of rows is greater than 3
                if len(buy_sell_period) > 3:
                    buy_sell_periods.append(buy_sell_period)
                buy_start = None  # Reset the buy_start after a sell has been found

    return buy_sell_periods


def plot_buy_sell_periods(buy_sell_periods, max_plots_per_row=5, max_rows_per_page=2):
    # Calculate total number of periods and number of plots per page
    num_periods = len(buy_sell_periods)
    plots_per_page = max_plots_per_row * max_rows_per_page

    # Split periods into chunks based on how many can fit on a page
    for page_start in range(0, num_periods, plots_per_page):
        # Determine the range of periods to plot on the current page
        page_end = min(page_start + plots_per_page, num_periods)
        num_plots = page_end - page_start

        # Calculate how many rows we need on this page
        num_rows = (
            num_plots + max_plots_per_row - 1
        ) // max_plots_per_row  # Ceiling division

        # Create subplots for this page
        fig, axes = plt.subplots(
            num_rows,
            max_plots_per_row,
            figsize=(20, 5 * num_rows),
            sharex=False,
            sharey=False,
        )
        axes = axes.flatten()  # Flatten axes array for easier iteration

        # Loop through the periods for this page and plot them
        for i, period in enumerate(buy_sell_periods[page_start:page_end]):
            # Plot the data for Low, Stop_profit, exit_price
            axes[i].plot(period["Date"], period["Low"], label="Low", marker="o")
            axes[i].plot(
                period["Date"], period["Stop_profit"], label="Stop_profit", marker="x"
            )
            axes[i].plot(
                period["Date"], period["exit_price"], label="exit_price", marker="^"
            )

            # Add the average cost (sAvgCost) as a red dashed line
            avg_cost = period["sAvgCost"].iloc[
                0
            ]  # Assuming sAvgCost is constant during the period
            if not pd.isna(avg_cost):
                axes[i].axhline(
                    y=avg_cost, color="red", linestyle="--", label="Avg Cost"
                )

            # Set labels and title
            sell_row = period.iloc[-1]  # Assuming last row in the period is the sell
            sProfit = sell_row["sProfit"] if not pd.isna(sell_row["sProfit"]) else "N/A"
            axes[i].set_title(
                f"Buy-Sell Period {page_start + i + 1} - Profit: {sProfit}%"
            )

            # Ensure the chart fills the space by scaling based on first and last data points
            axes[i].set_xlim([period["Date"].iloc[0], period["Date"].iloc[-1]])
            axes[i].set_ylim(
                [
                    min(period[["Low", "Stop_profit", "exit_price"]].min()),
                    max(period[["Low", "Stop_profit", "exit_price"]].max()),
                ]
            )

            # Format x-axis to show only first and last dates
            axes[i].set_xticks([period["Date"].iloc[0], period["Date"].iloc[-1]])
            axes[i].tick_params(axis="x", rotation=45)

            # Add legend
            axes[i].legend()

        # Remove unused subplots if any
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout for this page
        plt.tight_layout()

        # Show the plot for the current page
        plt.show()


if __name__ == "__main__":
    # 使用函数提取所有的buy-sell区间
    df = pd.read_csv(
        f"{REPORTS_DIR}/3867.KL1day_atr15bw15up15lw15_cut0.95pnl2ext3stp3.csv"
    )

    # df = df[df.sStatus != "Buy_filled"]
    # atr_cutoff_counts = df.apply(
    #     lambda row: (
    #         "ATR_Profit"
    #         if row["sStatus"] == "ATR_EXIT" and row["sProfit"] > 0
    #         else (
    #             "ATR_Loss"
    #             if row["sStatus"] == "ATR_EXIT" and row["sProfit"] < 0
    #             else row["sStatus"]
    #         )
    #     ),
    #     axis=1,
    # ).value_counts()

    # atr_percentages = atr_cutoff_counts / atr_cutoff_counts.sum()
    # result = pd.DataFrame(
    #     {"Count": atr_cutoff_counts, "Percentage": atr_percentages}
    # ).T.loc["Percentage"]
    # print(result.ATR_Profit)

    buy_sell_periods = split_buy_sell(df)
    plot_buy_sell_periods(buy_sell_periods, max_plots_per_row=7)
