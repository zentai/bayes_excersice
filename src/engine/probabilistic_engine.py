import sys
import os
from hunterverse.interface import IEngine
from utils import pandas_util
import numpy as np
import pandas as pd
from bayes import conditional, prob, odd, prob_odd, sigmoid, pmf_n_cdf, kde_top
from settings import ZERO

BAYESIAN_ENGINE_COLUMNS = [
    "Postrior",
    "Kelly",
    "p_win",
    "likelihood",
    "profit_margin",
    "loss_margin",
]


def x_times_odd(df, count):
    s_base = df.BuySignal == 1
    for i in range(count):
        s_base &= df.shift(-i).BuySignal == 1
    result_df = df[s_base]
    # print(result_df[['Date', 'BuySignal', 'profit', 'Kelly']][-60:])
    if len(result_df) == 0:
        return 1
    odd = (result_df.profit > 0).sum() / (
        (result_df.profit <= 0).sum() or 1 / (10**10)
    )
    return odd


def signal_meter(df, windows_size):
    return {i: x_times_odd(df, i) for i in range(1, windows_size)}


def pick_dates(df, today, windows):
    today = pd.to_datetime(today)
    df = df[["Date", "Matured"]].copy()
    trans_date = pd.to_datetime(df.Date)
    df.Matured = pd.to_datetime(df.Matured)
    date_interval = trans_date.diff().mode()[0].total_seconds()
    start_day = trans_date >= (today - pd.to_timedelta(windows * date_interval, "s"))
    s_matured = start_day & (df.Matured <= today)
    s_eod = start_day & (trans_date <= today)
    s_eod_yet_matured = (~s_matured) & s_eod
    return s_matured, s_eod, s_eod_yet_matured


def enrichment_temp_close(df):
    df = df.copy()
    last_day = df.iloc[-1]
    close = last_day.Close
    last_index = df.index[-1]
    s_sell_na = df.sell.isna()
    # # Create a dictionary to hold new column values
    # new_columns = {
    #     # "buy": close,
    #     "sell": close,
    #     "time_cost": list(reversed(range(len(df)))),
    #     "Matured": pd.to_datetime(last_day.Date),
    # }
    # # Update DataFrame in one go
    # for col, value in new_columns.items():
    #     df.loc[s_sell_na, col] = value
    df.loc[s_sell_na, "sell"] = close
    df["profit"] = (df["sell"] / df["buy"]) - 1
    return df


def enrichment_full_close(df):
    df = df.copy()
    last_day = df.iloc[-1]
    close = last_day.Close
    last_index = df.index[-1]
    df["buy"] = df["Close"]
    # Create a dictionary to hold new column values
    new_columns = {
        # "buy": close,
        "sell": close,
        "time_cost": list(reversed(range(len(df)))),
        "Matured": pd.to_datetime(last_day.Date),
    }
    # Update DataFrame in one go
    for col, value in new_columns.items():
        df[col] = value

    df["profit"] = (df["sell"] / df["buy"]) - 1
    return df


def kelly_formular(pwin, loss_margin, profit_margin):
    loss_margin = loss_margin or 1 / (10**30)
    profit_margin = profit_margin or 1 / (10**30)
    _pwin = pwin / loss_margin
    _ploss = (1 - pwin) / profit_margin
    # print(
    #     f"{_pwin:.6f} = {pwin:.6f}/{loss_margin:.6f} | {_ploss:.6f} = {(1 - pwin):.6f} / {profit_margin:.6f}"
    # )
    # if _pwin > _ploss:
    #     print(
    #         f"{(_pwin - _ploss) / _pwin if _pwin > _ploss else 0} = {_pwin } > {_ploss}"
    #     )
    return (_pwin - _ploss) / _pwin if _pwin > _ploss else 0


def calc_likelihood(signal_pnl, daily_pnl, n_mid):
    if len(daily_pnl) < n_mid or daily_pnl.isna().all():
        _like = 1
        p_win = 0.5
        profit_margin = 0
        loss_margin = 0
        return _like, p_win, profit_margin, loss_margin
    pmf, cdf = pmf_n_cdf(daily_pnl)
    profits = daily_pnl[daily_pnl > 0]
    loss = daily_pnl[daily_pnl <= 0]
    profit_margin = 0 if len(set(profits)) <= 1 else kde_top(profits)
    loss_margin = 0 if len(set(loss)) <= 1 else abs(loss.min())

    _zero = 1 / (10**n_mid)
    # signal_profit = max(len([i for i in signal_pnl if i > 0]), _zero)
    # signal_loss = max(len(signal_pnl) - signal_profit, _zero)
    # print(f' ==> P: {signal_profit}/{signal_loss} = {signal_profit/signal_loss}')

    # Calc likelihood
    # _, signal_cdf = pmf_n_cdf(signal_pnl)
    # _like = 1/signal_cdf(0) - 1   # profit_count/loss_count
    # w = sigmoid(len(signal_pnl), n_mid)
    # _pwin = 1 - signal_cdf(0)
    _like = 1 / (cdf(0) or 0.1) - (1 - _zero)  # profit_count/loss_count
    w = sigmoid(len(daily_pnl), n_mid)
    _pwin = 1 - cdf(0)

    return w * _like, _pwin, profit_margin, loss_margin


class BayesianEngine(IEngine):
    def __init__(self, params, name="default"):
        self._latest_prior = 0.5
        self._name = name
        self._params = params

    def hunt_plan(self, base_df):
        base_df = pandas_util.equip_fields(base_df, BAYESIAN_ENGINE_COLUMNS)
        windows = self._params.bayes_windows or 120
        df = base_df

        posterior = self._latest_prior
        # s_buysignal = (df.BuySignal == True) & (df.sell.isna())
        # if df.loc[df.Date == today, "Kelly"].isna().any():
        # for idx, row in df[df.BuySignal == True].iterrows():
        for idx, row in df[(df.BuySignal == True) & df.Kelly.isna()].iterrows():
            today = row.Date
            s_matured, s_eod, s_eod_yet_matured = pick_dates(df, today, windows)
            df_clone = df.loc[s_eod].copy()

            prior = 1
            if (df.BuySignal == 1).sum() > windows:
                profit_distribution = signal_meter(df[df.Date <= today], windows + 1)
                signal_count = 0
                for i in reversed(df_clone.BuySignal.values):
                    if i == 0:
                        break
                    signal_count += 1
                prior = profit_distribution.get(signal_count, 1)
                # print(
                #     f"[{today}] Prior: {profit_distribution.get(signal_count)}, SignalCount: {signal_count}, dist: {profit_distribution}"
                # )
            else:
                # print(f"!Data sample < {windows}")
                pass
            if s_eod_yet_matured.sum() > 0:
                sub_df = enrichment_temp_close(df[s_eod_yet_matured])
                df_clone.loc[
                    sub_df.index, ["sell", "time_cost", "Matured", "profit"]
                ] = sub_df[["sell", "time_cost", "Matured", "profit"]]
            daily_pnl = df_clone.profit

            # print(f"daily pnl count: {len(daily_pnl)}")
            # signal_pnl = df[(df.BuySignal == True) & (df.Date < today)][-windows:].profit
            signal_pnl = daily_pnl
            _like, p_win, profit_margin, loss_margin = calc_likelihood(
                signal_pnl, daily_pnl, windows / 2
            )

            posterior = prob_odd(prior * _like)

            kelly_args = {
                "pwin": posterior,
                "loss_margin": abs(loss_margin),
                "profit_margin": profit_margin,
            }

            _signal_kelly = kelly_formular(**kelly_args)

            if df.loc[df.Date == today, "Kelly"].isna().any():
                self._latest_prior = posterior
                # print(
                #     f"[{today}] {self._latest_prior}! {odd(posterior):.6f} * {_like:.6f} = {prob_odd(odd(posterior) * _like):.10f}"
                # )
                df.loc[df.Date == today, "Postrior"] = self._latest_prior
                df.loc[df.Date == today, "Kelly"] = _signal_kelly
                df.loc[df.Date == today, "p_win"] = p_win
                df.loc[
                    df.Date == today, "P/L"
                ] = f"{daily_pnl[daily_pnl>0].count()}/{daily_pnl[daily_pnl<=0].count()}"
                df.loc[df.Date == today, "likelihood"] = _like
                df.loc[df.Date == today, "profit_margin"] = profit_margin
                df.loc[df.Date == today, "loss_margin"] = loss_margin
            # print(f"{today} - {_signal_posterior} - {_signal_kelly}")
        return df
