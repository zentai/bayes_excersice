import sys
import os
from collections import namedtuple


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings import DATA_DIR, SRC_DIR, REPORTS_DIR

from bayes_opt import BayesianOptimization
from sklearn.model_selection import ParameterGrid
import functools

# from bayes_excercise import *
from bayes_kelly import enrichment_daily_profit
from bayes_kelly import s_turtle_buy
from bayes_kelly import pick_dates
from bayes_kelly import enrichment_temp_close
from bayes_kelly import calc_likelihood
from bayes_kelly import kelly_formular
from bayes_kelly import BayesKelly
from bayes_kelly import StrategyParam
from bayes_kelly import back_test
import matplotlib.pyplot as plt

from bayes import conditional, prob, odd, prob_odd
import numpy as np
import warnings

import pandas as pd
from empiricaldist import Pmf
import seaborn as sns
import numpy as np


def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    ax = plt.gca()
    ax.set(**options)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)

    plt.tight_layout()


def decorate_euro(title):
    decorate(xlabel="Profit/Loss (x)", ylabel="Probability", title=title)


def plot_profit_distribution(df):
    profit_data = df["profit"]
    profit_data = [round(x, 2) for x in profit_data]
    pmf = Pmf.from_seq(profit_data)

    cdf = pmf.make_cdf()
    cdf.plot(label="CDF")
    pmf.plot(label="PMF")

    # Find the 90% credible interval
    lower, upper = cdf.credible_interval(0.9)

    # Probability of no gains or loss and probability of earning a profit
    losses_zero = cdf.index[cdf.index <= 0].max()
    prob_loss = cdf(0)
    prob_profit = 1 - prob_loss

    # Add vertical dashed lines to represent the 90% credible interval
    plt.axvline(
        x=lower,
        color="red",
        linestyle="--",
        label=f"Lower Bound of 90% CI: {lower:.2f}",
    )
    plt.axvline(
        x=upper,
        color="green",
        linestyle="--",
        label=f"Upper Bound of 90% CI: {upper:.2f}",
    )
    plt.axvline(x=0, color="grey", linestyle=":", label=f"P(profit)={prob_profit:.2f}")

    # Annotate the plot with the 90% credible interval
    plt.annotate(
        f"Lower Bound of 90% CI: {lower:.2f}",
        xy=(lower, 0.05),
        xycoords="data",
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
        xytext=(0, 10),
        ha="center",
    )
    plt.annotate(
        f"Upper Bound of 90% CI: {upper:.2f}",
        xy=(upper, 0.95),
        xycoords="data",
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
        xytext=(0, 10),
        ha="center",
    )

    # Add the probabilities to labels
    print(
        f"The 90% credible interval for losses and gains is between {lower:.2f} and {upper:.2f}"
    )
    print(
        f"The Probability of no gains or loss is {prob_loss:.2f}, earn profit is {prob_profit:.2f}"
    )
    decorate_euro(title="Profit/Loss distribution")
    # plt.show()
    return pmf


# def plot_profit_distribution(df):
#     profit_data = df['profit']

#     # Calculate percentages
#     loss_percentage = (profit_data < 0).sum() / len(profit_data) * 100
#     zero_percentage = (profit_data == 0).sum() / len(profit_data) * 100
#     profit_percentage = (profit_data > 0).sum() / len(profit_data) * 100

#     profit_loss_ratio = profit_percentage / loss_percentage if loss_percentage != 0 else float('inf')

#     # Plot the KDE for PDF
#     sns.kdeplot(profit_data, label='Profit Distribution', fill=True)

#     plt.axvline(x=0, color='red', linestyle='dashed', linewidth=2)
#     plt.title('Profit Distribution')
#     plt.xlabel('Profit')
#     plt.ylabel('PDF')
#     plt.text(min(profit_data), plt.ylim()[1]*0.9, f'Loss: {loss_percentage:.2f}%', color='blue')
#     plt.text(0, plt.ylim()[1]*0.8, f'Zero: {zero_percentage:.2f}%', color='green')
#     plt.text(max(profit_data), plt.ylim()[1]*0.9, f'Profit: {profit_percentage:.2f}%', color='blue')
#     plt.text(max(profit_data), plt.ylim()[1]*0.7, f'Profit/Loss Ratio: {profit_loss_ratio:.2f}', color='purple')
#     plt.grid(True)
#     plt.legend()
#     plt.show()


if __name__ == "__main__":

    def load_and_split_data(code, train_ratio=0.4):
        df = pd.read_csv(f"{DATA_DIR}/{code}.csv")
        df = df.dropna()
        size = len(df)
        train_size = int(train_ratio * size)
        train_df = df[:train_size]
        test_df = df[train_size:]
        return train_df, test_df

    # code = 'P9D.SI'
    # train_df, test_df = load_and_split_data(code)
    # best_params = {'ATR_sample': 1966.033839067926, 'atr_loss_margin': 4.134997288129421, 'bayes_windows': 221.92100262598285, 'lower_sample': 828.2296397115283, 'upper_sample': 265.18110976011883}
    # sp = StrategyParam(**best_params)

    # code = 'NVDA'
    # train_df, test_df = load_and_split_data(code)
    # best_params = {'ATR_sample': 1966.033839067926, 'atr_loss_margin': 4.134997288129421, 'bayes_windows': 221.92100262598285, 'lower_sample': 828.2296397115283, 'upper_sample': 265.18110976011883}
    # best_params = {'ATR_sample': 1626.3750027860333, 'atr_loss_margin': 4.9767908336249915, 'bayes_windows': 1279.2608970048873, 'lower_sample': 271.7235871266356, 'upper_sample': 1583.848549540803}
    # best_params = {'ATR_sample': 689.3873618253018, 'atr_loss_margin': 4.700160536947254, 'bayes_windows': 251.11130023195366, 'lower_sample': 584.0122340471805, 'upper_sample': 770.1933545844585}
    # sp = StrategyParam(**best_params)

    # print(sp)
    # bkf = BayesKelly(train_df, sp)
    # bkf.register_signal('TurtleBuy', s_turtle_buy)
    # kelly_df = bkf.bayes_update(prior=0.5, debug=True)
    # profit_table, w_daily_return, simulate_transaction_df = back_test(kelly_df, breakdown=False)
    # print(profit_table)
    # print(w_daily_return)
    # print(simulate_transaction_df)
    # mix_1 = plot_profit_distribution(kelly_df)

    code = "BTC-USD"
    train_df, test_df = load_and_split_data(code)
    best_params = {
        "ATR_sample": 526.3434480988084,
        "atr_loss_margin": 3.9707438480375874,
        "bayes_windows": 59.984676812368576,
        "lower_sample": 895.1654132411534,
        "upper_sample": 254.47909372327248,
    }
    sp = StrategyParam(**best_params)

    # print(sp)
    # bkf = BayesKelly(train_df, sp)
    # bkf.register_signal('TurtleBuy', s_turtle_buy)
    # kelly_df = bkf.bayes_update(prior=0.5, debug=True)
    # profit_table, w_daily_return, simulate_transaction_df = back_test(kelly_df, breakdown=False)
    # print(profit_table)
    # print(w_daily_return)
    # print(simulate_transaction_df)
    # mix_2 = plot_profit_distribution(kelly_df)

    # code = 'E28.SI'
    # train_df, test_df = load_and_split_data(code)
    # best_params = {'ATR_sample': 526.3434480988084, 'atr_loss_margin': 3.9707438480375874, 'bayes_windows': 59.984676812368576, 'lower_sample': 895.1654132411534, 'upper_sample': 254.47909372327248}
    # best_params = {'ATR_sample': 526.2986739995059, 'atr_loss_margin': 3.9707438480375874, 'bayes_windows': 59.9200954851833, 'lower_sample': 895.86688860857146, 'upper_sample': 254.8939252997294}
    # best_params = {'ATR_sample': 78.0, 'atr_loss_margin': 3.9, 'bayes_windows': 674.7708553788615, 'lower_sample': 705.9319165629884, 'upper_sample': 432.0875159711735}
    # sp = StrategyParam(**best_params)

    # code = '6888.KL'
    # train_df, test_df = load_and_split_data(code)
    # best_params = {'ATR_sample': 900.0, 'atr_loss_margin': 5.0, 'bayes_windows': 345.65941919872034, 'lower_sample': 2.0, 'upper_sample': 319.12734685706664}
    # sp = StrategyParam(**best_params)

    # code = '5148.KL'
    # train_df, test_df = load_and_split_data(code)
    # best_params = {'ATR_sample': 552.0, 'atr_loss_margin': 3.361, 'bayes_windows': 199.0, 'lower_sample': 888.0, 'upper_sample': 195.0}
    # sp = StrategyParam(**best_params)

    print(sp)
    bkf = BayesKelly(test_df, sp)
    bkf.register_signal("TurtleBuy", s_turtle_buy)
    kelly_df = bkf.bayes_update(prior=0.5, debug=True)
    profit_table, w_daily_return, simulate_transaction_df = back_test(
        kelly_df, breakdown=True
    )
    print(profit_table)
    print(w_daily_return)
    print(simulate_transaction_df)
    plot_profit_distribution(kelly_df)
    plt.show()
