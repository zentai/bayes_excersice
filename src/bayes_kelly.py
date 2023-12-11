import sys
import os
from icecream import ic

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from settings import DATA_DIR, SRC_DIR, REPORTS_DIR

import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import functools
from bayes import conditional, prob, odd, prob_odd, sigmoid, pmf_n_cdf
from settings import ZERO
from empiricaldist import Pmf

terminal_view = [
    "Date",
    "Open",
    "Close",
    "buy",
    "sell",
    "time_cost",
    "Stop_profit",
    "profit",
    "Posterior",
    "kelly_f",
    "MaxDrawdown",
    "MeanProfit",
    "Profit.kelly_f",
    "Invest",
    "Income",
    "Total_fund",
]
real_time_view = [
    "Date",
    "Open",
    "Close",
    "buy",
    "sell",
    "time_cost",
    "Stop_profit",
    "profit",
    "Posterior",
    "kelly_f",
    "MaxDrawdown",
    "MeanProfit",
]


def turtle_trading(base_df, params):
    # Initialize columns if they don't exist
    for col in ["ATR", "turtle_h", "turtle_l", "Stop_profit"]:
        base_df[col] = base_df.get(col, np.nan)

    # performance: only re-calc nessasary part.
    # start_idx = base_df.ATR.isna().idxmax()
    idx = (
        base_df.index
        if base_df.ATR.isna().all()
        else base_df.ATR.iloc[params.ATR_sample :].isna().index
    )

    base_df.loc[idx, "turtle_h"] = (
        base_df.Close.shift(1).rolling(params.upper_sample).max()
    )
    base_df.loc[idx, "turtle_l"] = (
        base_df.Close.shift(1).rolling(params.lower_sample).min()
    )
    base_df.loc[idx, "h_l"] = base_df.High - base_df.Low
    base_df.loc[idx, "c_h"] = (base_df.Close.shift(1) - base_df.High).abs()
    base_df.loc[idx, "c_l"] = (base_df.Close.shift(1) - base_df.Low).abs()
    base_df.loc[idx, "TR"] = base_df[["h_l", "c_h", "c_l"]].max(axis=1)
    base_df.loc[idx, "ATR"] = base_df["TR"].rolling(params.ATR_sample).mean()
    base_df.loc[idx, "Stop_profit"] = (
        base_df.Close.shift(1) - base_df.ATR.shift(1) * params.atr_loss_margin
    )
    base_df.loc[idx, "exit_price"] = base_df[["turtle_l", "Stop_profit"]].max(axis=1)
    return base_df


def s_turtle_buy(base_df, params):
    df = turtle_trading(base_df, params)
    return df.Close > df.turtle_h


def enrichment_daily_profit(base_df, params):
    _loss_margin = params.atr_loss_margin or 1.5
    base_df = turtle_trading(base_df, params)

    # Initialize columns if they don't exist
    for col in ["buy", "sell", "profit", "time_cost", "Matured"]:
        base_df[col] = base_df.get(col, pd.NaT if col == "Matured" else np.nan)

    resume_idx = base_df.sell.isna().idxmax()
    df = base_df.loc[resume_idx:].copy()

    # Buy daily basic when the market price is higher than Stop profit
    buy_condition = (
        df.buy.isna() & df.Stop_profit.notna() & (df.Stop_profit < df.Open.shift(-1))
    )
    df.loc[buy_condition, "buy"] = df.Open.shift(-1)

    # Sell condition:
    sell_condition = df.buy.notna() & (
        (df.Close.shift(-1) < df.Stop_profit) | (df.Close.shift(-1) < df.turtle_l)
    )

    df.loc[sell_condition, "sell"] = df.Stop_profit.where(sell_condition)
    df.loc[sell_condition, "Matured"] = pd.to_datetime(
        df.Date.shift(-1).where(sell_condition)
    )

    # Backfill sell and Matured columns
    df.sell.bfill(inplace=True)
    df.Matured.bfill(inplace=True)

    # Compute profit and time_cost columns
    profit_condition = df.buy.notna() & df.sell.notna() & df.profit.isna()
    df.loc[profit_condition, "profit"] = (df.sell / df.buy) - 1
    df.loc[profit_condition, "time_cost"] = pd.to_datetime(df.Matured) - pd.to_datetime(
        df.Date
    )

    # Clear sell and Matured values where buy is NaN
    df.loc[df.buy.isna(), ["sell", "Matured"]] = np.nan
    base_df.update(df)
    return base_df


def calc_annual_return_and_sortino_ratio(cost, profit, df):
    _yield_curve_1yr = 0.0419
    _start_date = df.iloc[0].Date
    _end_date = df.iloc[-1].Date
    _trade_count = len(df[df.kelly_f > 0])
    _trade_minutes = (
        pd.to_datetime(_end_date) - pd.to_datetime(_start_date)
    ).total_seconds() / 60
    _annual_trade_count = (_trade_count / _trade_minutes) * 365 * 24 * 60
    _downside_risk_stdv = df[
        (df.kelly_f > 0) & (df.profit < _yield_curve_1yr)
    ].profit.std(ddof=1)
    _annual_downside_risk_stdv = _downside_risk_stdv * np.sqrt(_annual_trade_count) or 1
    t = _trade_minutes / (365 * 24 * 60)
    _annual_return = (profit / cost) ** (1 / t) - 1 if (profit / cost > 0) else 0
    _sortino_ratio = (_annual_return - _yield_curve_1yr) / _annual_downside_risk_stdv
    return _annual_return, _sortino_ratio


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


def kelly_formular(pwin, loss_margin, profit_margin):
    loss_margin = loss_margin or ZERO
    profit_margin = profit_margin or ZERO
    _pwin = pwin / loss_margin
    _ploss = (1 - pwin) / profit_margin
    return (_pwin - _ploss) / _pwin if _pwin > _ploss else 0


from dataclasses import dataclass, field


@dataclass
class StrategyParam:
    ATR_sample: int = 20
    atr_loss_margin: float = 1.0
    bayes_windows: int = 120
    lower_sample: int = 10
    upper_sample: int = 20

    def __post_init__(self):
        self.ATR_sample = int(self.ATR_sample)
        self.bayes_windows = int(self.bayes_windows)
        self.lower_sample = int(self.lower_sample)
        self.upper_sample = int(self.upper_sample)


class BayesKelly:
    def __init__(self, df, params, name="default"):
        self._df = enrichment_daily_profit(df, params)
        self._book = {}
        self._signals = {}
        self._signals_posterior = {}
        self._latest_prior = 0.5
        self._name = name
        self._params = params

    def register_signal(self, name, preproccess_func):
        s_signal = preproccess_func(self._df, self._params)
        self._signals.update({name: s_signal})
        self._signals_posterior.update({name: 0.5})
        dates = self._df.loc[s_signal, "Date"].values

        for d in dates:
            self._book.setdefault(d, []).append((name, s_signal))

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

    def bayes_update(self, prior=0.5, debug=False):
        windows = self._params.bayes_windows or 120
        self._df.loc[:, "Signal"] = False
        _prior = odd(prior)
        _rs = []
        _desc = []
        for today in sorted(self._book.keys()):
            df = self._df.copy()
            s_matured, s_eod, s_eod_yet_matured = pick_dates(df, today, windows)
            sub_df = enrichment_temp_close(df[s_eod_yet_matured])
            df.loc[sub_df.index, ["sell", "time_cost", "Matured", "profit"]] = sub_df[
                ["sell", "time_cost", "Matured", "profit"]
            ]
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

            s_mix_signal = self._df.Signal
            profits = df["profit"]
            N_mid = max(windows * 0.01, 3)
            _like, p_win, profit_margin, loss_margin = calc_likelihood(profits, N_mid)

            _max_drawdown = (
                abs(-1)
                if np.isnan(df[s_loss].profit.min())
                else abs(df[s_loss].profit.min())
            )
            _mean_profit = (
                0
                if np.isnan(df[s_profit].profit.mean())
                else df[s_profit].profit.mean()
            )

            _kelly_f = kelly_formular(
                pwin=prob_odd(_posterior),
                loss_margin=_max_drawdown,
                profit_margin=_mean_profit,
            )
            # print(f'[{today}] kelly(pwin={prob_odd(_posterior)}, loss={_max_drawdown}, porifit={_mean_profit}) = {_kelly_f}')

            # log
            _i = max(df.loc[df.Date == today].index)
            # we should using self._df instead of df, because df using T close for temp close position price. that should be update daily.
            _fundamental = list(
                self._df.loc[
                    _i, ["Close", "buy", "sell", "time_cost", "profit", "Matured"]
                ].values
            )
            _rs.append(
                [today]
                + _fundamental
                + [prob_odd(_prior)]
                + a
                + [_like, prob_odd(_posterior), _kelly_f, _max_drawdown, _mean_profit]
            )

            # update
            _prior = _posterior

        _kelly_df = pd.DataFrame(
            _rs,
            columns=[
                "Date",
                "Close",
                "buy",
                "sell",
                "time_cost",
                "profit",
                "Matured",
                "Prior",
                "ttl_name",
                "ttl_w",
                "ttl_like",
                "ttl_profit_count",
                "ttl_loss_count",
                "bbl_name",
                "bbl_w",
                "bbl_like",
                "bbl_profit_count",
                "bbl_loss_count",
                "Likelihood",
                "Posterior",
                "kelly_f",
                "MaxDrawdown",
                "MeanProfit",
            ],
        )
        if debug:
            # _kelly_df.to_csv(f'{REPORTS_DIR}/{self._name}_dynamic_kelly.csv', index=False)
            # print(f'Build: {REPORTS_DIR}/{self._name}_dynamic_kelly.csv')
            # self._signal_center.plot_performance()
            pass

        # migrate to original history table
        # full_df = pd.merge(df, _bk_df[['Date', 'Posterior', 'kelly_f']], how='left', on=['Date'])
        return _kelly_df


def s_monthly_buy(base_df):
    return base_df.Date.apply(
        lambda x: x.split("-")[-1] == "01" or x.split("-")[-1] == "05"
    )
    # return base_df.Date.apply(lambda x: True)


def s_bollinger_band(base_df):
    bbh_window = int(tparam.get("bbh_window", 5))
    bbl_window = int(tparam.get("bbl_window", 5))
    bbl_grade = int(tparam.get("bbl_grade", 5))
    bollinger_config = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    df = base_df[["Date", "Close"]].copy()

    def boll_statis(df, col_name, bollinger_config, mv):
        boll_df = pd.DataFrame()
        for std_ratio in bollinger_config:
            if col_name == "BBL":
                shifted = df.Close.shift(1).rolling(mv)
                boll_df[str(std_ratio)] = df.Close <= (
                    shifted.mean() - (shifted.std() * std_ratio)
                )
            elif col_name == "BBH":
                shifted = df.Close.shift(1).rolling(mv)
                boll_df[str(std_ratio)] = df.Close >= (
                    shifted.mean() + (shifted.std() * std_ratio)
                )
        df[col_name] = boll_df.sum(axis=1)
        return df

    boll_statis(df, "BBL", bollinger_config, bbh_window)
    boll_statis(df, "BBH", bollinger_config, bbl_window)
    return df.BBH > bbl_grade


def s_vegas_tunnel_buy(base_df):
    df = base_df[["Date", "Close"]].copy()
    # 计算均线
    df["12EMA"] = df["Close"].ewm(span=12).mean()
    df["144EMA"] = df["Close"].ewm(span=144).mean()
    df["169EMA"] = df["Close"].ewm(span=169).mean()
    df["576EMA"] = df["Close"].ewm(span=576).mean()
    df["676EMA"] = df["Close"].ewm(span=676).mean()

    # 计算斐波那契位置
    df["fib1"] = df["144EMA"] + 0.618 * (df["169EMA"] - df["144EMA"])
    df["fib2"] = df["144EMA"] + 1.618 * (df["169EMA"] - df["144EMA"])
    df["fib3"] = df["144EMA"] + 2.618 * (df["169EMA"] - df["144EMA"])
    df["fib4"] = df["144EMA"] + 4.236 * (df["169EMA"] - df["144EMA"])
    df["fib5"] = df["144EMA"] + 6.854 * (df["169EMA"] - df["144EMA"])

    # 进入隧道区间的信号
    df["in_tunnel"] = np.where(
        (df["Close"] > df["12EMA"]) & (df["Close"] < df["144EMA"]), True, False
    )

    # 突破隧道的信号
    df["breakout"] = np.where(
        (df["Close"] > df["144EMA"]) & (df["12EMA"] > df["144EMA"]), True, False
    )

    # 突破上轨和下轨的信号
    df["long_signal"] = np.where(
        (df["breakout"] == True) & (df["Close"] > df["fib1"]), True, False
    )
    df["short_signal"] = np.where(
        (df["breakout"] == True) & (df["Close"] < df["fib1"]), True, False
    )

    return df.long_signal


def weighted_daily_return(df_subset):
    total_time_cost = df_subset["time_cost"].sum() or 1
    weighted_return = np.sum(
        (1 + df_subset["profit"]) ** (1 / df_subset["time_cost"])
        * df_subset["time_cost"]
    )
    wd_return = (weighted_return / total_time_cost) - 1
    return total_time_cost, wd_return


def back_test(base_df, initial_fund=1000, breakdown=False):
    max_invest = (
        initial_fund
        if np.isnan(tparam.get("max_invest"))
        else int(tparam.get("max_invest"))
    )
    r = []
    total_fund = initial_fund
    breakdown_rows = []
    future_profits = {}

    df = base_df.copy()
    df.fillna(0, inplace=True)

    for today, _close, _profit, _time_cost, _matured_date, pct in df[
        ["Date", "Close", "profit", "time_cost", "Matured", "kelly_f"]
    ].values:
        d = pd.to_datetime(today)
        # fetch profit for Matured date deals.
        matured_date = [fd for fd in future_profits.keys() if fd <= d]
        today_profit = sum([sum(future_profits.pop(d)) for d in matured_date])

        total_fund = total_fund + today_profit
        investable_fund = total_fund
        # Only invest when we have enough fund.
        if total_fund >= 10:
            _max_invest = min(max_invest, total_fund)
            # Calculate bet amount
            _invest, _keep = _max_invest * pct, _max_invest * (1 - pct)
            if _invest:
                # register future profit
                if _time_cost > 0:
                    future_date = _matured_date
                    if future_date not in future_profits:
                        future_profits[future_date] = []
                    future_profit = _invest * (1 + _profit)
                    future_profits[future_date].append(future_profit)
                else:  # Unknown profit yet
                    future_date = np.nan
                    future_profit = np.nan
                    _profit = np.nan
            else:
                # No invest due to no confidence
                _invest = 0
                future_date = np.nan
                future_profit = np.nan
                _profit = np.nan
            total_fund -= _invest
        else:
            # No invest due to out of money
            _invest = 0
            future_date = np.nan
            future_profit = np.nan
            _profit = np.nan

        future_date = future_date
        future_profit = future_profit

        breakdown_rows.append(
            [
                today,
                _close,
                today_profit,
                investable_fund,
                _invest,
                _profit,
                future_profit,
                future_date,
                total_fund,
            ]
        )

    # added unclaim Matured profit back if any.
    if future_profits:
        profit_date = [fd for fd in future_profits.keys()]

        today_profit = 1
        d = np.nan
        for fd in profit_date:
            d = fd
            f_profit = sum(future_profits.pop(fd))
            today_profit += f_profit
        total_fund = total_fund + today_profit
        breakdown_rows.append(
            [
                d,
                np.nan,
                today_profit,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                total_fund,
            ]
        )

    sample = len(df)
    profit_sample = len(df[df.profit > 0])
    # profit_mean = (df.profit > 0).mean()
    # loss_mean = (df.profit < 0).mean() or ZERO
    # profit_loss_ratio = profit_mean / loss_mean

    profit_data = [round(x, 2) for x in df["profit"]]
    pmf = Pmf.from_seq(profit_data)
    pmf.normalize()
    cdf = pmf.make_cdf()

    # Probability of no gains or loss and probability of earning a profit
    prob_loss = cdf(0)
    prob_win = 1 - prob_loss

    _date = pd.to_datetime(df["Date"])
    total_years = (_date.max() - _date.min()).days / 365.25
    w = sigmoid(sample, total_years)
    profit_loss_ratio = w * prob_win

    cost = initial_fund
    profit = total_fund - cost
    time_cost, w_daily_return = weighted_daily_return(df)
    avg_time_cost = time_cost / sample if sample else 0
    avg_profit = profit / sample if sample else 0
    drawdown = df.profit.min()

    _annual_return, _sortino_ratio = calc_annual_return_and_sortino_ratio(
        cost, profit, df
    )
    r.append(
        [
            profit,
            cost,
            profit_loss_ratio,
            sample,
            profit_sample,
            avg_time_cost,
            avg_profit,
            drawdown,
            _annual_return,
            _sortino_ratio,
        ]
    )
    profit_table = pd.DataFrame(
        r,
        columns=[
            "Profit",
            "Cost",
            "ProfitLossRatio",
            "Sample",
            "ProfitSample",
            "Avg.Timecost",
            "Avg.Profit",
            "Drawdown",
            "Annual.Return",
            "SortinoRatio",
        ],
    )
    bdown_df = pd.DataFrame(
        breakdown_rows,
        columns=[
            "Date",
            "Close",
            "Income",
            "CashOnHand",
            "Invest",
            "FutureProfit",
            "FutProfitAmount",
            "FuturePaydate",
            "Total_fund",
        ],
    )
    bdown_df = pd.merge(
        df[["Date", "Posterior", "kelly_f", "MaxDrawdown", "MeanProfit"]],
        bdown_df,
        how="outer",
        on=["Date"],
    )
    bdown_df["Profit.kelly_f"] = (bdown_df["FutureProfit"] > 0) & (
        bdown_df["kelly_f"] > 0
    )
    if breakdown:
        bdown_df.to_csv(f"{REPORTS_DIR}/simulate_transaction.csv", index=False)
        print(f"Build: {REPORTS_DIR}/simulate_transaction.csv")

    return (
        profit_table,
        w_daily_return,
        bdown_df[
            [
                "Date",
                "Posterior",
                "kelly_f",
                "Profit.kelly_f",
                "MaxDrawdown",
                "MeanProfit",
                "Invest",
                "FutProfitAmount",
                "FuturePaydate",
                "Income",
                "Total_fund",
            ]
        ],
    )


def calc_confidence_betratio(base_df, signal_func, prior=0.5, debug=False):
    # Refactor suggestion: Extract the enrichment of daily profit into a separate function
    # Reason: This will make the code more modular and easier to test
    base_df = base_df
    windows = int(tparam.get("bayes_windows", 120))
    _prior = odd(prior)
    _rs = []
    _desc = []

    sell_idx = base_df[np.isnan(base_df["sell"])].index
    sell_windows = (
        pd.to_datetime(base_df.iloc[-1].Date)
        - pd.to_datetime(base_df.loc[sell_idx[0]].Date)
    ).days

    is_scratch = "kelly_f" not in base_df.columns
    if is_scratch:
        # Refactor suggestion: Extract the assignment of new columns into a separate function
        # Reason: This will make the code more readable and easier to maintain
        base_df = base_df.assign(
            Prior=np.nan,
            Likelihood=np.nan,
            Posterior=np.nan,
            kelly_f=np.nan,
            MaxDrawdown=np.nan,
            MeanProfit=np.nan,
        )

    windows = max(windows, sell_windows)
    mv_window = len(base_df) if is_scratch else windows
    df = base_df.iloc[-mv_window:].copy()

    s_signal = signal_func(df)

    if is_scratch:
        bayes_idx = df[s_signal].index
    else:
        mask = ~np.isnan(df.Posterior)
        _prior = df.loc[mask, "Posterior"].iloc[-1]
        _prior = odd(_prior)
        bayes_idx = df[s_signal].loc[np.isnan(df.Posterior)].index

    # Refactor suggestion: Extract the calculation of prior, posterior, likelihood, max_drawdown, mean_profit, and kelly into separate functions
    # Reason: This will make the code more modular, easier to test, and easier to understand
    prior = []
    posterior = []
    like = []
    max_drawdown = []
    mean_profit = []
    kelly = []

    for today in df.loc[bayes_idx].Date:
        s_matured, s_eod, s_eod_yet_matured = pick_dates(df, today, windows)
        sub_df = enrichment_temp_close(df[s_eod_yet_matured])
        df.loc[sub_df.index, ["sell", "time_cost", "Matured", "profit"]] = sub_df[
            ["sell", "time_cost", "Matured", "profit"]
        ]

        s_profit = s_eod & (df.profit > 0)
        s_loss = s_eod & ~s_profit
        w_sub_signal = [0.9]

        s_mix_signal = s_signal
        _like, debug_list = calc_likelihood(s_profit, s_loss, s_mix_signal)
        _posterior = _prior * _like
        _max_drawdown = (
            abs(-1)
            if np.isnan(df[s_loss].profit.min())
            else abs(df[s_loss].profit.min())
        )
        _mean_profit = (
            0
            if np.isnan(df[s_profcalc_likelihoodit].profit.mean())
            else df[s_profit].profit.mean()
        )
        _kelly_f = kelly_formular(
            pwin=prob_odd(_posterior),
            loss_margin=_max_drawdown,
            profit_margin=_mean_profit,
        )

        prior.append(prob_odd(_prior))
        like.append(_like)
        posterior.append(prob_odd(_posterior))
        kelly.append(_kelly_f)
        max_drawdown.append(_max_drawdown)
        mean_profit.append(_mean_profit)

        _prior = _posterior
        df = base_df.iloc[-mv_window:].copy()

    # Refactor suggestion: Extract the assignment of new values to base_df into a separate function
    # Reason: This will make the code more readable and easier to maintain
    base_df.loc[bayes_idx, "Prior"] = prior
    base_df.loc[bayes_idx, "Likelihood"] = like
    base_df.loc[bayes_idx, "Posterior"] = posterior
    base_df.loc[bayes_idx, "MaxDrawdown"] = max_drawdown
    base_df.loc[bayes_idx, "MeanProfit"] = mean_profit
    base_df.loc[bayes_idx, "kelly_f"] = kelly

    return base_df


tparam = {
    "ATR_sample": 116,
    "atr_loss_margin": 2.00000,
    "bayes_windows": 198,
    "lower_sample": 113,
    "upper_sample": 8,
    "max_invest": 100,
}


# Function to create profit bins and distribution with custom bins
def create_profit_bins(df):
    # Define separate bins for negative and positive profit values
    negative_bins = np.linspace(df["profit"].min(), 0, 50)
    positive_bins = np.linspace(0, df["profit"].max(), 50)
    bins = np.concatenate([negative_bins, positive_bins[1:]])

    # Cut the 'profit' column using the defined bins
    df["profit_bins"] = pd.cut(df["profit"], bins=bins, labels=False)

    # Calculate the frequency distribution of the bins
    profit_distribution = df["profit_bins"].value_counts().sort_index()

    # Plot the distribution
    profit_distribution.plot(kind="bar", figsize=(12, 6))
    plt.title("Profit Distribution (Separate Bins for Negative and Positive Values)")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()

    return df, profit_distribution


if __name__ == "__main__":
    # TODO: Consider moving these settings to a separate settings module
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    pd.set_option("display.width", 300)

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from settings import DATA_DIR, SRC_DIR, REPORTS_DIR

    code = "BTC-USD"
    df = pd.read_csv(f"{DATA_DIR}/{code}.csv")
    df = df.dropna()
    size = len(df)

    start_idx = 0
    windows = 200
    # TODO: Consider creating a function to handle the creation of the base dataframe
    base_df = df[:1024].copy()

    # best_params = {
    #     "ATR_sample": 526.3434480988084,
    #     "atr_loss_margin": 3.9707438480375874,
    #     "bayes_windows": 59.984676812368576,
    #     "lower_sample": 895.1654132411534,
    #     "upper_sample": 254.47909372327248,
    # }

    best_params = {
        "ATR_sample": 10,
        "atr_loss_margin": 1,
        "bayes_windows": 10,
        "lower_sample": 10,
        "upper_sample": 10,
    }
    sp = StrategyParam(**best_params)
    ic("Original DF: ", base_df, sp)
    bkf = BayesKelly(base_df, sp)

    base_df = bkf._df
    for i in range(10):
        new_data = df.iloc[1024 + i]
        base_df = pd.concat(
            [base_df, pd.DataFrame([new_data], columns=base_df.columns)]
        )

        bkf = BayesKelly(base_df, sp)
        ic(bkf._df[-59:])
        ic(len(bkf._df))

    # bkf.register_signal("TurtleBuy", s_turtle_buy)
    # kelly_df = bkf.bayes_update(prior=0.5, debug=True)

    # base_df = calc_confidence_betratio(base_df, s_turtle_buy)
    # last_row = df.iloc[-1].Date
    # print(base_df)
    # base_df.to_csv(f"{REPORTS_DIR}/{code}_signal.csv", index=False)
    # print(f"{REPORTS_DIR}/{code}_signal.csv")
    # create_profit_bins(base_df)
    # TODO: Consider creating a function to handle the loop for updating the base dataframe
    # for i in range(size):
    #     new_data = df.iloc[start_idx+windows+i]
    #     base_df = base_df.append(new_data)
    #     base_df = calc_confidence_betratio(base_df, s_turtle_buy)
    #     if last_row == base_df.iloc[-1].Date:
    #         break

    # profit_table, w_daily_return, simulate_transaction_df = back_test(base_df, breakdown=False)
    # print(profit_table)
    # merge_backtest = True
    # if merge_backtest:
    #     print(base_df.tail(10))
    #     # TODO: Consider creating a function to handle the merging and saving of dataframes
    #     base_df = pd.merge(base_df, simulate_transaction_df[['Date', 'Profit.kelly_f', 'Invest', 'Income', 'Total_fund']], how='left', on=['Date'])
    #     base_df.to_csv(f'{REPORTS_DIR}/{code}_backtest.csv', index=False)
    #     print(f'created: reports/{code}_backtest.csv')
    # else:
    #     base_df.to_csv(f'{REPORTS_DIR}/{code}_signal.csv', index=False)
    #     print(f'created: reports/{code}_signal.csv')

    # print(base_df[terminal_view].tail(60))
