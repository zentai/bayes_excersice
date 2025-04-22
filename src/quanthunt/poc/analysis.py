import matplotlib.pyplot as plt
import pandas as pd
from config import config

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_average_cumulative_return(
    filenames, date_column="Date", price_column="Close", start_date=None, end_date=None
):
    """
    繪製多支股票的平均累積回報率隨時間變化的圖表。

    參數：
    - filenames：列表，包含股票歷史數據的 CSV 檔案路徑。
    - date_column：字符串，日期列的列名（預設為 'Date'）。
    - price_column：字符串，價格列的列名（預設為 'Close'）。
    - start_date：字符串或 None，分析的開始日期（包含）。
    - end_date：字符串或 None，分析的結束日期（包含）。
    """

    cumulative_returns = []

    for filename in filenames:
        # 讀取 CSV 檔案
        df = pd.read_csv(filename)

        # 確保日期列為 datetime 類型
        df[date_column] = pd.to_datetime(df[date_column])

        # 按日期排序
        df = df.sort_values(by=date_column)

        # 篩選日期範圍
        if start_date is not None:
            df = df[df[date_column] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df[date_column] <= pd.to_datetime(end_date)]

        # 重置索引
        df = df.reset_index(drop=True)

        # 計算每日回報率
        df["Return"] = df[price_column].pct_change()

        # 計算累積回報率
        df["Cumulative Return"] = (1 + df["Return"]).cumprod()

        # 儲存日期和累積回報率
        cumulative_returns.append(df[[date_column, "Cumulative Return"]])

    # 合併所有股票的累積回報率
    merged_returns = pd.DataFrame()

    for i, cr in enumerate(cumulative_returns):
        cr = cr.rename(columns={"Cumulative Return": f"Cumulative Return {i}"})
        if merged_returns.empty:
            merged_returns = cr
        else:
            merged_returns = pd.merge(merged_returns, cr, on=date_column, how="outer")

    # 設置日期為索引
    merged_returns = merged_returns.sort_values(by=date_column)
    merged_returns = merged_returns.set_index(date_column)

    # 向前填充缺失值
    merged_returns = merged_returns.fillna(method="ffill")

    # 刪除仍有 NaN 的行
    merged_returns = merged_returns.dropna()

    # Calculate average cumulative return
    merged_returns["Average Cumulative Return"] = merged_returns.mean(axis=1)

    # Plot the chart
    plt.figure(figsize=(12, 6))
    plt.plot(
        merged_returns.index,
        merged_returns["Average Cumulative Return"],
        label="Average Cumulative Return",
        linewidth=2,
    )

    # Enhance the chart with proper labels and titles
    plt.title("Average Cumulative Return Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.legend(prop={"size": 12})
    plt.grid(True)
    plt.tight_layout()

    # 顯示圖表
    plt.show()


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


def market_trend():
    # stock_list = "0018.KL,1023.KL,3026.KL,4677.KL,5099.KL,5200.KL,5299.KL,7033.KL,8567.KL,0023.KL,1066.KL,3336.KL,4707.KL,5109.KL,5202.KL,5318.KL,7084.KL,8583.KL,0041.KL,1155.KL,3417.KL,4715.KL,5148.KL,5204.KL,5347.KL,7106.KL,8869.KL,0072.KL,1171.KL,3476.KL,5008.KL,5168.KL,5205.KL,5819.KL,7113.KL,0078.KL,1295.KL,3689.KL,5014.KL,5175.KL,5209.KL,6012.KL,7123.KL,0082.KL,1562.KL,3743.KL,5020.KL,5182.KL,5239.KL,6076.KL,7153.KL,0138.KL,1651.KL,3794.KL,5038.KL,5183.KL,5246.KL,6556.KL,7164.KL,0146.KL,1724.KL,3816.KL,5053.KL,5184.KL,5264.KL,6742.KL,7183.KL,0166.KL,2445.KL,3867.KL,5062.KL,5185.KL,5279.KL,6888.KL,7216.KL,0216.KL,2828.KL,4065.KL,5075.KL,5196.KL,5285.KL,6947.KL,7293.KL".split(
    #     ","
    # )
    stock_list = "1A4.SI,5GZ.SI,A7RU.SI,BSL.SI,C6L.SI,G13.SI,M35.SI,P40U.SI,S7OU.SI,U96.SI,1B1.SI,5IF.SI,AIY.SI,BTOU.SI,C76.SI,G92.SI,M44U.SI,P9D.SI,S7P.SI,U9E.SI,1C0.SI,5IG.SI,AJ2.SI,BUOU.SI,CC3.SI,H02.SI,ME8U.SI,Q0X.SI,T14.SI,V03.SI,1D0.SI,5TP.SI,AJBU.SI,BVA.SI,CRPU.SI,H13.SI,N2IU.SI,Q5T.SI,T82U.SI,Y06.SI,1D4.SI,5TT.SI,AWX.SI,BWCU.SI,D03.SI,H20.SI,NR7.SI,RE4.SI,TQ5.SI,Y92.SI,1J5.SI,5UX.SI,AZI.SI,C06.SI,D05.SI,H30.SI,NS8U.SI,S41.SI,TS0U.SI,Z25.SI,40T.SI,9CI.SI,B61.SI,C07.SI,E5H.SI,H78.SI,O10.SI,S58.SI,U09.SI,Z59.SI,558.SI,9I7.SI,BDA.SI,C09.SI,F34.SI,I07.SI,O39.SI,S59.SI,U11.SI,Z74.SI,5AB.SI,A17U.SI,BJZ.SI,C2PU.SI,F9D.SI,J69U.SI,O5RU.SI,S63.SI,U13.SI,5CP.SI,A30.SI,BN4.SI,C38U.SI,FQ7.SI,K71U.SI,OV8.SI,S68.SI,U14.SI,5G1.SI,A50.SI,BS6.SI,C52.SI,G07.SI,M1GU.SI,P15.SI,S71.SI,U77.SI".split(
        ","
    )

    stock_list = [f"{DATA_DIR}/{file}_cached.csv" for file in stock_list]
    plot_average_cumulative_return(stock_list)


if __name__ == "__main__":

    # market_trend()
    df = pd.read_csv(
        f"{REPORTS_DIR}/500.SI1day_atr15bw15up15lw15_cut0.9pnl2ext3stp5.csv"
    )

    buy_sell_periods = split_buy_sell(df)
    plot_buy_sell_periods(buy_sell_periods, max_plots_per_row=7)
