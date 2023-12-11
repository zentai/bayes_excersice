import os
from datetime import datetime
import pandas as pd
import sys
import matplotlib.pyplot as plt


def monte_carlo_simulation(dataframe, num_simulations, num_trades_per_simulation):
    """
    Perform Monte Carlo simulations on a trading strategy.

    Parameters:
    - dataframe: DataFrame containing 'profit' and 'kelly(f)' columns.
    - num_simulations: Number of simulations to perform.
    - num_trades_per_simulation: Number of trades in each simulation.

    Outputs:
    - CSV files for each simulation.
    - Plot of capital growth trajectories.
    - Statistics including average profit-loss ratio, win rate, average capital growth, best and worst capital growth.
    """
    print(dataframe)

    initial_capital = 1000  # Initial capital
    stats = {"average_profit_loss_ratio": [], "win_rate": [], "final_capital": []}

    # Create a folder for the output with a timestamp
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # folder_name = f"MonteCarloSimulation_{timestamp}"
    # os.makedirs(folder_name, exist_ok=True)

    # Run the simulations
    for i in range(num_simulations):
        shuffled_data = (
            dataframe[["profit", "kelly(f)"]]
            .sample(n=num_trades_per_simulation, replace=True)
            .reset_index(drop=True)
        )
        capital = initial_capital
        capital_trajectory = [capital]

        for _, row in shuffled_data.iterrows():
            bet_amount = capital * row["kelly(f)"]
            capital += bet_amount * row["profit"]
            capital_trajectory.append(capital)

            if capital <= 0:  # Stop simulation if capital falls to zero
                capital = 0
                break

        # Save the shuffled data to a CSV file
        # shuffled_data.to_csv(f"{folder_name}/simulation_{i+1}.csv", index=False)

        # Plotting the capital trajectory for this simulation
        plt.plot(capital_trajectory)

        # Collecting statistics
        profits = shuffled_data[shuffled_data["profit"] > 0]["profit"]
        losses = shuffled_data[shuffled_data["profit"] < 0]["profit"].abs()
        average_profit = profits.mean() if not profits.empty else 0
        average_loss = losses.mean() if not losses.empty else 0
        profit_loss_ratio = average_profit / average_loss if average_loss != 0 else 0
        win_rate = len(profits) / len(shuffled_data) * 100

        stats["average_profit_loss_ratio"].append(profit_loss_ratio)
        stats["win_rate"].append(win_rate)
        stats["final_capital"].append(capital_trajectory[-1])

    # Final plot adjustments
    plt.title("Capital Growth Trajectories for Monte Carlo Simulations")
    plt.xlabel("Number of Trades")
    plt.ylabel("Capital")
    plt.show()

    # Calculating final statistics
    average_profit_loss_ratio = sum(stats["average_profit_loss_ratio"]) / len(
        stats["average_profit_loss_ratio"]
    )
    average_win_rate = sum(stats["win_rate"]) / len(stats["win_rate"])
    average_final_capital = sum(stats["final_capital"]) / len(stats["final_capital"])
    best_final_capital = max(stats["final_capital"])
    worst_final_capital = min(stats["final_capital"])

    return {
        "average_profit_loss_ratio": average_profit_loss_ratio,
        "average_win_rate": average_win_rate,
        "average_final_capital": average_final_capital,
        "best_final_capital": best_final_capital,
        "worst_final_capital": worst_final_capital,
    }


# Note: This function can now be called with a DataFrame, number of simulations, and number of trades per simulation.
# Example usage:
# results = monte_carlo_simulation(dataframe=data, num_simulations=10, num_trades_per_simulation=50)
# This will run 10 simulations with 50 trades each and output the results and CSV files.

if __name__ == "__main__":
    # TODO: Consider moving these settings to a separate settings module
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    pd.set_option("display.width", 300)

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from settings import DATA_DIR, SRC_DIR, REPORTS_DIR

    code = "BTC-USD"
    df = pd.read_csv(f"{REPORTS_DIR}/backup/ZKS-USD.csv_best_kelly.csv")
    size = len(df)

    results = monte_carlo_simulation(
        dataframe=df, num_simulations=2000, num_trades_per_simulation=100
    )
    from pprint import pprint as pp

    pp(results)
