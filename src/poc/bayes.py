import pandas as pd
from empiricaldist import Pmf
import numpy as np
from scipy.stats import gaussian_kde
from config import config
ZERO = config.zero

def update(table):
    table["unnorm"] = table["prior"] * table["likelihood"]
    prob_data = table["unnorm"].sum()
    table["posterior"] = table["unnorm"] / prob_data
    return prob_data


def prob(A):
    return A.mean()


def conditional(proposition, given):
    return prob(proposition[given])


def odd(p):
    p = 0.999999 if p == 1 else p
    return p / (1 - p + 1e-10)


def prob_odd(o):
    return o / (o + 1 + 1e-10)


def sigmoid(x, N_mid=50):
    """
    Sigmoid function to weight the sample count.

    Parameters:
    - x: sample count
    - N_mid: Midpoint for sigmoid function adjustment

    Returns:
    - Weight: A value between 0 and 1
    """
    return 1 / (1 + np.exp(-(x - N_mid)))


def pmf_n_cdf(data_list):
    data_list = [round(x, 5) for x in data_list]
    pmf = Pmf.from_seq(data_list)
    pmf.normalize()
    cdf = pmf.make_cdf()
    return pmf, cdf


def profit_dist_by_CDF(data_list, n_mid):
    pmf, cdf = pmf_n_cdf(data_list)
    lower, upper = cdf.credible_interval(0.9)
    positive_profits = {x: pmf[x] for x in pmf.qs if x > 0}
    profit_margin = (
        max(positive_profits, key=positive_profits.get) if positive_profits else ZERO
    )
    loss_margin = abs(lower)
    prob_loss = cdf(0)
    prob_win = 1 - prob_loss

    # Calc likelihood
    _like = prob_win / max(prob_loss, ZERO)
    w = sigmoid(len(data_list), n_mid)

    return w * _like, prob_win, profit_margin, loss_margin


def kde_top(profits):
    values = profits.to_numpy()
    kde = gaussian_kde(values)
    x = np.linspace(min(values), max(values), 1000)
    pdf = kde.evaluate(x)
    max_density_value = x[np.argmax(pdf)]
    return max_density_value
