import sys
import os
from story import IEngine
from utils import pandas_util
import numpy as np
import pandas as pd
from bayes import conditional, prob, odd, prob_odd, sigmoid, pmf_n_cdf


def pick_dates(df, today, windows):
    today = pd.to_datetime(today)
    df = df[["Date", "Matured"]].copy()
    trans_date = pd.to_datetime(df.Date)
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

    # Create a dictionary to hold new column values
    new_columns = {
        "buy": close,
        "sell": close,
        "time_cost": list(reversed(range(len(df)))),
        "Matured": pd.to_datetime(last_day.Date),
    }

    # Update DataFrame in one go
    for col, value in new_columns.items():
        if isinstance(value, list):
            df[col] = value
        else:
            df.at[last_index, col] = value

    df["profit"] = (df["sell"] / df["buy"]) - 1

    return df


def calc_likelihood(profits, n_mid):
    pmf, cdf = pmf_n_cdf(profits)
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
    w = sigmoid(len(profits), n_mid)

    return w * _like, prob_win, profit_margin, loss_margin


class BayesianEngine(IEngine):
    def __init__(self, params, name="default"):
        self._latest_prior = 0.5
        self._name = name
        self._params = params

    def generate_hunt_plan(self, recon_report):
        windows = self._params.bayes_windows or 120
        df = recon_report
        df.loc[:, "Signal"] = False
        prior = 0.5
        _prior = odd(prior)
        _rs = []
        _desc = []
        _posterior
        for idx, row in df[df.BuySignal == True].iterrows():
            today = row.Date
            # print(idx, today, row.BuySignal)
            s_matured, s_eod, s_eod_yet_matured = pick_dates(df, today, windows)

            sub_df = enrichment_temp_close(df[s_eod_yet_matured])
            df_clone = df.loc[s_eod].copy()
            df_clone.update(sub_df[["sell", "time_cost", "Matured", "profit"]])

            profits = df_clone.profit
            N_mid = max(windows * 0.01, 3)
            _like, p_win, profit_margin, loss_margin = calc_likelihood(profits, N_mid)

            _signal_prior = odd(_posterior)
            _signal_posterior = prob_odd(_signal_prior * _like)

            kelly_args = {
                "pwin": _signal_posterior,
                "loss_margin": loss_margin,
                "profit_margin": profit_margin,
            }

            _signal_kelly = kelly_formular(**kelly_args)
            self._signals_posterior[name] = _signal_posterior
            return _signal_posterior, _signal_kelly, profit_margin * _signal_posterior

            # Noted. after using temp close solution, we should calculate all eod trades instead of matured.
            s_profit = s_eod & (df.profit > 0)
            s_loss = s_eod & ~s_profit

            # Single Strategy
            a = [""] * 10
            for name, s_signal in self._book.get(today):
                _signal_posterior, _signal_kelly, exp_profit = self.sub_signal(
                    name, today
                )
                _posterior = _signal_posterior

                self._df.loc[self._df.Date == today, "Signal"] = True
                self._df.loc[self._df.Date == today, "Source"] = name
                self._df.loc[self._df.Date == today, "SignalW"] = _signal_kelly
                self._df.loc[self._df.Date == today, "SignalExpectProfit"] = exp_profit
                self._df.loc[self._df.Date == today, "SignalCnt"] = len(
                    self._book.get(today)
                )

    def sub_signal(self, name, today):
        windows = self._params.bayes_windows or 120

        # We are going to using temp close solution
        s_matured, s_eod, s_eod_yet_matured = pick_dates(self._df, today, windows)

        df, sub_df = self._df.loc[s_eod].copy(), enrichment_temp_close(
            self._df.loc[s_eod_yet_matured]
        )

        df.update(sub_df[["sell", "time_cost", "Matured", "profit"]])

        profits = df["profit"]
        N_mid = max(windows * 0.01, 3)
        _like, p_win, profit_margin, loss_margin = calc_likelihood(profits, N_mid)

        _signal_prior = odd(self._signals_posterior.get(name, 0))
        _signal_posterior = prob_odd(_signal_prior * _like)

        kelly_args = {
            "pwin": _signal_posterior,
            "loss_margin": loss_margin,
            "profit_margin": profit_margin,
        }

        _signal_kelly = kelly_formular(**kelly_args)
        self._signals_posterior[name] = _signal_posterior
        return _signal_posterior, _signal_kelly, profit_margin * _signal_posterior

    #         s_mix_signal = self._df.Signal
    #         profits = df["profit"]
    #         N_mid = max(windows * 0.01, 3)
    #         _like, p_win, profit_margin, loss_margin = calc_likelihood(profits, N_mid)

    #         _max_drawdown = (
    #             abs(-1)
    #             if np.isnan(df[s_loss].profit.min())
    #             else abs(df[s_loss].profit.min())
    #         )
    #         _mean_profit = (
    #             0
    #             if np.isnan(df[s_profit].profit.mean())
    #             else df[s_profit].profit.mean()
    #         )

    #         _kelly_f = kelly_formular(
    #             pwin=prob_odd(_posterior),
    #             loss_margin=_max_drawdown,
    #             profit_margin=_mean_profit,
    #         )
    #         # print(f'[{today}] kelly(pwin={prob_odd(_posterior)}, loss={_max_drawdown}, porifit={_mean_profit}) = {_kelly_f}')

    #         # log
    #         _i = max(df.loc[df.Date == today].index)
    #         # we should using self._df instead of df, because df using T close for temp close position price. that should be update daily.
    #         _fundamental = list(
    #             self._df.loc[
    #                 _i, ["Close", "buy", "sell", "time_cost", "profit", "Matured"]
    #             ].values
    #         )
    #         _rs.append(
    #             [today]
    #             + _fundamental
    #             + [prob_odd(_prior)]
    #             + a
    #             + [_like, prob_odd(_posterior), _kelly_f, _max_drawdown, _mean_profit]
    #         )

    #         # update
    #         _prior = _posterior

    #     _kelly_df = pd.DataFrame(
    #         _rs,
    #         columns=[
    #             "Date",
    #             "Close",
    #             "buy",
    #             "sell",
    #             "time_cost",
    #             "profit",
    #             "Matured",
    #             "Prior",
    #             "ttl_name",
    #             "ttl_w",
    #             "ttl_like",
    #             "ttl_profit_count",
    #             "ttl_loss_count",
    #             "bbl_name",
    #             "bbl_w",
    #             "bbl_like",
    #             "bbl_profit_count",
    #             "bbl_loss_count",
    #             "Likelihood",
    #             "Posterior",
    #             "kelly_f",
    #             "MaxDrawdown",
    #             "MeanProfit",
    #         ],
    #     )
    #     if debug:
    #         # _kelly_df.to_csv(f'{REPORTS_DIR}/{self._name}_dynamic_kelly.csv', index=False)
    #         # print(f'Build: {REPORTS_DIR}/{self._name}_dynamic_kelly.csv')
    #         # self._signal_center.plot_performance()
    #         pass

    #     # migrate to original history table
    #     # full_df = pd.merge(df, _bk_df[['Date', 'Posterior', 'kelly_f']], how='left', on=['Date'])
    #     return _kelly_df

    # def bayes_update(self, prior=0.5, debug=False):
    #     windows = self._params.bayes_windows or 120
    #     self._df.loc[:, "Signal"] = False
    #     _prior = odd(prior)
    #     _rs = []
    #     _desc = []
    #     for today in sorted(self._book.keys()):
    #         df = self._df.copy()
    #         s_matured, s_eod, s_eod_yet_matured = pick_dates(df, today, windows)
    #         sub_df = enrichment_temp_close(df[s_eod_yet_matured])
    #         df.loc[sub_df.index, ["sell", "time_cost", "Matured", "profit"]] = sub_df[
    #             ["sell", "time_cost", "Matured", "profit"]
    #         ]
    #         # Noted. after using temp close solution, we should calculate all eod trades instead of matured.
    #         s_profit = s_eod & (df.profit > 0)
    #         s_loss = s_eod & ~s_profit

    #         # Single Strategy
    #         a = [""] * 10
    #         for name, s_signal in self._book.get(today):
    #             _signal_posterior, _signal_kelly, exp_profit = self.sub_signal(
    #                 name, today
    #             )
    #             _posterior = _signal_posterior

    #             self._df.loc[self._df.Date == today, "Signal"] = True
    #             self._df.loc[self._df.Date == today, "Source"] = name
    #             self._df.loc[self._df.Date == today, "SignalW"] = _signal_kelly
    #             self._df.loc[self._df.Date == today, "SignalExpectProfit"] = exp_profit
    #             self._df.loc[self._df.Date == today, "SignalCnt"] = len(
    #                 self._book.get(today)
    #             )

    #         s_mix_signal = self._df.Signal
    #         profits = df["profit"]
    #         N_mid = max(windows * 0.01, 3)
    #         _like, p_win, profit_margin, loss_margin = calc_likelihood(profits, N_mid)

    #         _max_drawdown = (
    #             abs(-1)
    #             if np.isnan(df[s_loss].profit.min())
    #             else abs(df[s_loss].profit.min())
    #         )
    #         _mean_profit = (
    #             0
    #             if np.isnan(df[s_profit].profit.mean())
    #             else df[s_profit].profit.mean()
    #         )

    #         _kelly_f = kelly_formular(
    #             pwin=prob_odd(_posterior),
    #             loss_margin=_max_drawdown,
    #             profit_margin=_mean_profit,
    #         )
    #         # print(f'[{today}] kelly(pwin={prob_odd(_posterior)}, loss={_max_drawdown}, porifit={_mean_profit}) = {_kelly_f}')

    #         # log
    #         _i = max(df.loc[df.Date == today].index)
    #         # we should using self._df instead of df, because df using T close for temp close position price. that should be update daily.
    #         _fundamental = list(
    #             self._df.loc[
    #                 _i, ["Close", "buy", "sell", "time_cost", "profit", "Matured"]
    #             ].values
    #         )
    #         _rs.append(
    #             [today]
    #             + _fundamental
    #             + [prob_odd(_prior)]
    #             + a
    #             + [_like, prob_odd(_posterior), _kelly_f, _max_drawdown, _mean_profit]
    #         )

    #         # update
    #         _prior = _posterior

    #     _kelly_df = pd.DataFrame(
    #         _rs,
    #         columns=[
    #             "Date",
    #             "Close",
    #             "buy",
    #             "sell",
    #             "time_cost",
    #             "profit",
    #             "Matured",
    #             "Prior",
    #             "ttl_name",
    #             "ttl_w",
    #             "ttl_like",
    #             "ttl_profit_count",
    #             "ttl_loss_count",
    #             "bbl_name",
    #             "bbl_w",
    #             "bbl_like",
    #             "bbl_profit_count",
    #             "bbl_loss_count",
    #             "Likelihood",
    #             "Posterior",
    #             "kelly_f",
    #             "MaxDrawdown",
    #             "MeanProfit",
    #         ],
    #     )
    #     if debug:
    #         # _kelly_df.to_csv(f'{REPORTS_DIR}/{self._name}_dynamic_kelly.csv', index=False)
    #         # print(f'Build: {REPORTS_DIR}/{self._name}_dynamic_kelly.csv')
    #         # self._signal_center.plot_performance()
    #         pass

    #     # migrate to original history table
    #     # full_df = pd.merge(df, _bk_df[['Date', 'Posterior', 'kelly_f']], how='left', on=['Date'])
    #     return _kelly_df
