import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import functools
from bayes import conditional, prob, odd, prob_odd

terminal_view = ['Date', 'Open', 'Close', 'buy', 'sell', 'time_cost', 'Stop_profit', 'profit', 'Posterior', 'kelly_f', 'MaxDrawdown', 'MeanProfit', 'Profit.kelly_f', 'Invest', 'Income', 'Total_fund']


def turtle_trading(base_df):
    upper_sample = int(tparam.get('upper_sample', 20) )
    lower_sample = int(tparam.get('lower_sample', 10) )
    ATR_sample = int(tparam.get('ATR_sample', 20) )

    is_scratch = 'ATR' not in base_df.columns
    windows = len(base_df) if is_scratch else np.max([upper_sample, lower_sample, ATR_sample]) + 1

    # performance: only re-calc nessasary part.
    df = base_df.iloc[-windows:].copy()
    idx = df.index if is_scratch else df[np.isnan(df['ATR'])].index
    df = df.assign(turtle_h = df.Close.shift(1).rolling(upper_sample).max())
    df = df.assign(turtle_l = df.Close.shift(1).rolling(lower_sample).min())
    df = df.assign(h_l = df.High - df.Low)
    df = df.assign(c_h = (df.Close.shift(1)-df.High).abs())
    df = df.assign(c_l = (df.Close.shift(1)-df.Low).abs())
    df = df.assign(TR = df[['h_l', 'c_h', 'c_l']].max(axis=1))
    df = df.assign(ATR = (df.TR.rolling(ATR_sample).sum()/ATR_sample))

    # copy value to base_df
    base_df.loc[idx, 'turtle_h'] = df.loc[idx, 'turtle_h']
    base_df.loc[idx, 'turtle_l'] = df.loc[idx, 'turtle_l']
    base_df.loc[idx, 'ATR'] = df.loc[idx, 'ATR']
    return base_df


def s_turtle_buy(base_df):
    df = turtle_trading(base_df)
    return df.Close > df.turtle_h


def enrichment_daily_profit(base_df):
    _loss_margin = tparam.get('atr_loss_margin', 1.5)
    base_df = turtle_trading(base_df)

    sell = []
    time_cost = []

    is_scratch = 'buy' not in base_df.columns
    if is_scratch:
        base_df = base_df.assign(buy=np.nan)
        base_df = base_df.assign(sell=np.nan)
        base_df = base_df.assign(time_cost=np.nan)

    sell_idx = base_df[np.isnan(base_df['sell'])].index
    windows = -len(base_df) if is_scratch else sell_idx[0]

    df = base_df.loc[windows:].copy()
    buy_idx = df.index if is_scratch else df[np.isnan(df['buy'])].index
    df.loc[buy_idx, 'buy'] = df.Open.shift(-1).loc[buy_idx]

    stop_profits = []

    for i, _v in enumerate(df.loc[sell_idx][['buy', 'sell', 'time_cost', 'ATR']].values):
        _buy, _sell, _time_cost, _buy_atr = _v
        _pre_atr = _buy_atr
        _pre_close = _buy
        stop_profit = _pre_close - _pre_atr * _loss_margin
        if np.isnan(_buy):
            sell.append(_sell)
            stop_profits.append(stop_profit)
            time_cost.append(_time_cost)
            continue
        if not np.isnan(_sell):
            sell.append(_sell)
            stop_profits.append(stop_profit)
            time_cost.append(_time_cost)
            continue

        sell_point = None
        days = 0
        for j, v in enumerate(df[['Close', 'turtle_l', 'ATR']].iloc[i+1:].values):
            _close, _turtle_low, _atr = v
            stop_profit = _pre_close - _pre_atr * _loss_margin
            sell_point, days = (_close, j) if (_close < stop_profit) or (_close < _turtle_low) else (None, None)
            if sell_point:
                break
            _pre_close = _close
            _pre_atr = _atr
        if sell_point:
            sell.append(sell_point)
            stop_profits.append(stop_profit)
            time_cost.append(days+1)
        else:
            sell.append(np.nan)
            stop_profits.append(stop_profit)
            time_cost.append(np.nan)

    # print(f'{i} - {len(buy_idx)} - {len(sell_idx)} - {len(sell)}')
    df.loc[sell_idx, 'sell'] = sell
    df.loc[sell_idx, 'time_cost'] = time_cost
    df.loc[sell_idx, 'Stop_profit'] = stop_profits
    df.loc[sell_idx, 'Matured'] = pd.to_datetime(df.loc[sell_idx, 'Date']) + pd.to_timedelta(df.loc[sell_idx, 'time_cost'], 'm')
    df.loc[sell_idx, 'profit'] = (df.loc[sell_idx, 'sell'] / df.loc[sell_idx, 'buy']) - 1

    base_df.loc[buy_idx, 'buy'] = df.loc[buy_idx, 'buy']
    base_df.loc[sell_idx, 'sell'] = df.loc[sell_idx, 'sell']
    base_df.loc[sell_idx, 'time_cost'] = df.loc[sell_idx, 'time_cost']
    base_df.loc[sell_idx, 'Matured'] = df.loc[sell_idx, 'Matured']
    base_df.loc[sell_idx, 'Stop_profit'] = df.loc[sell_idx, 'Stop_profit']
    base_df.loc[sell_idx, 'profit'] = df.loc[sell_idx, 'profit']
    return base_df


def calc_annual_return_and_sortino_ratio(cost, profit, df, interval):
    interval_minutes = {
        '1min': 1,
        '5min': 5,
        '15min': 15,
        '30min': 30,
        '1hour': 60,
        '4hour': 240,
        '1day': 1440
    }

    _yield_curve_1yr = 0.0419
    _start_date = df.iloc[0].Date
    _end_date = df.iloc[-1].Date
    _trade_count = len(df)
    _trade_minutes = (pd.to_datetime(_end_date) - pd.to_datetime(_start_date)).total_seconds() / 60
    _interval_trade_minutes = _trade_minutes * interval_minutes[interval]
    _annual_trade_count = (_trade_count / _interval_trade_minutes) * 365 * 24 * 60
    _downside_risk_stdv = df[df.profit < _yield_curve_1yr].profit.std(ddof=1)
    _annual_downside_risk_stdv = _downside_risk_stdv * np.sqrt(_annual_trade_count)
    t = _interval_trade_minutes / (365 * 24 * 60)
    _annual_return = (profit / cost) ** (1/t) - 1 if (profit / cost > 0) else 0
    _sortino_ratio = (_annual_return - _yield_curve_1yr) / _annual_downside_risk_stdv
    return _annual_return, _sortino_ratio


def enrichment_temp_close(df, today):
    df = df.copy()
    close = df.at[df.index[-1], 'Close']
    df.at[df.index[-1], 'buy'] = close      # we always take next day open as buy, but since we remove T+1, so we consider no profit at the last day
    df = df.assign(sell=close)
    df = df.assign(time_cost = (pd.to_datetime(today) - pd.to_datetime(df['Date'])).dt.days)
    df = df.assign(Matured = pd.to_datetime(today))
    df = df.assign(profit = (df.sell / df.buy) - 1)
    return df

def adjust_like(w, _like):
    if w >= 1 or _like == 1:    # w >= 1 means sample enough, _like == 1 mean nothing change.
        return _like
    return (_like - 1) * w + 1 if (_like > 1) else 1-((1 - _like) * w)    # adjust _like size by w.

def calc_likelihood(s_profit, s_loss, s_signal):
    # within the window, how many profit/loss sample (not consider signal)
    _proft_count = max(s_profit.sum(), 0.001)
    _loss_count = max(s_loss.sum(), 0.001)
    w = min((_proft_count + _loss_count)/30, 1)

    # signal within the window, win rate.
    s_signal_profit = s_profit & s_signal
    s_signal_loss = s_loss & s_signal
    _signal_proft_count = max(s_signal_profit.sum(), 0.99999)
    _signal_loss_count = max(s_signal_loss.sum(), 0.99999)
    _like = _signal_proft_count / _signal_loss_count
    debug_list = [w, _like, _signal_proft_count, _signal_loss_count]
    return adjust_like(w, _like), [adjust_like(w, _like)]+debug_list    # adjust _like size by w.


def pick_dates(df, today, windows):
    today = pd.to_datetime(today)
    df = df[['Date', 'Matured']].copy()
    df['Date'] = pd.to_datetime(df.Date)
    start_day = df.Date >= (today - pd.to_timedelta(windows, 'm'))
    s_matured = start_day & (df.Matured <= today)
    s_eod = start_day & (df.Date <= today)
    s_eod_yet_matured = (~s_matured) & s_eod
    return s_matured, s_eod, s_eod_yet_matured


def kelly_formular(pwin, loss_margin, profit_margin):
    loss_margin = loss_margin or 0.00001
    profit_margin = profit_margin or 0.00001
    _pwin = (pwin / loss_margin)
    _ploss = ((1 - pwin) / profit_margin)
    return (_pwin - _ploss) / _pwin if _pwin > _ploss else 0


class BayesKelly:

    def __init__(self, df, name='default'):
        self._df = enrichment_daily_profit(df)
        self._book = {}
        self._signals = {}
        self._signals_posterior = {}
        self._latest_prior = 0.5
        self._name = name

    def register_signal(self, name, preproccess_func):
        s_signal = preproccess_func(self._df)
        self._signals[name] = s_signal
        self._signals_posterior[name] = 0.5
        dates = self._df[s_signal].Date.values
        for d in dates:
            if d not in self._book:
                self._book[d] = []
            self._book[d].append((name, s_signal))

    def sub_signal(self, name, today):
        windows = int(tparam.get('bayes_windows', 120))

        # clone df, we are going to using temp close solution
        df = self._df.copy()
        s_matured, s_eod, s_eod_yet_matured = pick_dates(df, today, windows)
        df = df[s_eod]
        sub_df = enrichment_temp_close(df[s_eod_yet_matured], today)

        # full settlement df
        df.loc[sub_df.index, ['sell', 'time_cost', 'Matured', 'profit']] = sub_df[['sell', 'time_cost', 'Matured', 'profit']]

        # s_profit/s_loss in rolling window
        s_profit = s_eod & (df.profit > 0)
        s_loss = s_eod & ~s_profit

        #  keep independence posterior calculation
        s_signal = self._signals[name]
        _signal_prior = odd(self._signals_posterior.get(name))
        _like, debug_list = calc_likelihood(s_profit, s_loss, s_signal)
        _signal_posterior = prob_odd(_signal_prior * _like)
        _signal_mean_profit = 0 if np.isnan(df[s_profit].profit.mean()) else df[s_profit].profit.mean()
        _signal_max_drawdown = -1 if np.isnan(df[s_loss].profit.min()) else df[s_loss].profit.min()
        _signal_kelly = kelly_formular(pwin=_signal_posterior, loss_margin=abs(_signal_max_drawdown), profit_margin=_signal_mean_profit)
        self._signals_posterior[name] = _signal_posterior

        today_profit = df[df.Date==today].profit.values[0]
        # print(f'mean profit: {df.profit.mean()} - {df.profit.mean()*_signal_posterior}， profit: {today_profit}')
        return _signal_posterior, _signal_kelly, df.profit.mean()*_signal_posterior, debug_list

    def bayes_update(self, prior=0.5, debug=False):
        windows = int(tparam.get('bayes_windows', 120))
        self._df.loc[:, 'Signal'] = False
        _prior = odd(prior)
        _rs = []
        _desc = []
        for today in sorted(self._book.keys()):
            df = self._df.copy()
            s_matured, s_eod, s_eod_yet_matured = pick_dates(df, today, windows)
            sub_df = enrichment_temp_close(df[s_eod_yet_matured], today)
            df.loc[sub_df.index, ['sell', 'time_cost', 'Matured', 'profit']] = sub_df[['sell', 'time_cost', 'Matured', 'profit']]
            # Noted. after using temp close solution, we should calculate all eod trades instead of matured.
            s_profit = s_eod & (df.profit > 0)
            # s_profit = s_matured & (df.profit > 0)

            s_loss = s_eod & ~s_profit
            w_sub_signal = []
            for name, s_signal in self._book.get(today):

                _signal_posterior, _signal_kelly, exp_profit, debug_list = self.sub_signal(name, today)
                w_sub_signal.append(0.9 if name == 'TurtleBuy' else 0.1)

                # only take valueable buy signal.
                # if _signal_kelly > 0:
                #     # print(f'add {today} signal: {name} due to Kelly: {_signal_kelly}')
                #     self._df.loc[self._df.Date == today, 'Signal'] = True
                self._df.loc[self._df.Date == today, 'Signal'] = True
                self._df.loc[self._df.Date == today, 'Source'] = name
                self._df.loc[self._df.Date == today, 'SignalW'] = _signal_kelly
                self._df.loc[self._df.Date == today, 'SignalDebug'] = str(debug_list)
                self._df.loc[self._df.Date == today, 'SignalExpectProfit'] = exp_profit
                self._df.loc[self._df.Date == today, 'SignalCnt'] = len(self._book.get(today))


            # ok, start to recalce new signal
            s_mix_signal = self._df.Signal
            _like, debug_list = calc_likelihood(s_profit, s_loss, s_mix_signal)

            a = [''] * 10
            a[0], a[1], a[2], a[3], a[4], a[5] = ['MixBuy'] + debug_list
            _posterior = _prior * adjust_like(sum(w_sub_signal), _like)
            # _posterior = _prior * _like * np.mean(w_sub_signal)

            _max_drawdown = abs(-1) if np.isnan(df[s_loss].profit.min()) else abs(df[s_loss].profit.min())
            _mean_profit = 0 if np.isnan(df[s_profit].profit.mean()) else df[s_profit].profit.mean()

            _kelly_f = kelly_formular(pwin=prob_odd(_posterior), loss_margin=_max_drawdown, profit_margin=_mean_profit)
            # print(f'[{today}] kelly(pwin={prob_odd(_posterior)}, loss={_max_drawdown}, porifit={_mean_profit}) = {_kelly_f}')

            # log
            _i = max(df.loc[df.Date==today].index)
            # we should using self._df instead of df, because df using T close for temp close position price. that should be update daily.
            _fundamental = list(self._df.loc[_i, ['Close', 'buy', 'sell', 'time_cost', 'profit', 'Matured']].values)
            _rs.append([today] + _fundamental + [prob_odd(_prior)] + a + [_like, prob_odd(_posterior), _kelly_f, _max_drawdown, _mean_profit])

            # update
            _prior = _posterior

        _kelly_df = pd.DataFrame(_rs, columns=['Date', 'Close', 'buy', 'sell', 'time_cost', 'profit', 'Matured', 'Prior', 'ttl_name', 'ttl_w', 'ttl_like', 'ttl_profit_count','ttl_loss_count', 'bbl_name', 'bbl_w', 'bbl_like', 'bbl_profit_count','bbl_loss_count', 'Likelihood', 'Posterior', 'kelly_f', 'MaxDrawdown', 'MeanProfit'])
        if debug:
            _kelly_df.to_csv(f'reports/{self._name}_dynamic_kelly.csv', index=False)
            print(f'Build: reports/{self._name}_dynamic_kelly.csv')
            # self._signal_center.plot_performance()

        # migrate to original history table
        # full_df = pd.merge(df, _bk_df[['Date', 'Posterior', 'kelly_f']], how='left', on=['Date'])
        return _kelly_df





def s_monthly_buy(base_df):
    return base_df.Date.apply(lambda x: x.split('-')[-1]=='01' or x.split('-')[-1]=='05')
    # return base_df.Date.apply(lambda x: True)


def s_bollinger_band(base_df):
    bbh_window = int(tparam.get('bbh_window', 5))
    bbl_window = int(tparam.get('bbl_window', 5))
    bbl_grade = int(tparam.get('bbl_grade', 5))
    bollinger_config = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    df = base_df[['Date', 'Close']].copy()

    def boll_statis(df, col_name, bollinger_config, mv):
        boll_df = pd.DataFrame()
        for std_ratio in bollinger_config:
            if col_name == 'BBL':
                shifted = df.Close.shift(1).rolling(mv)
                boll_df[str(std_ratio)] = df.Close <= (shifted.mean() - (shifted.std() * std_ratio))
            elif col_name == 'BBH':
                shifted = df.Close.shift(1).rolling(mv)
                boll_df[str(std_ratio)] = df.Close >= (shifted.mean() + (shifted.std() * std_ratio))
        df[col_name] = boll_df.sum(axis=1)
        return df

    boll_statis(df, 'BBL', bollinger_config, bbh_window)
    boll_statis(df, 'BBH', bollinger_config, bbl_window)
    return df.BBH > bbl_grade


def s_vegas_tunnel_buy(base_df):
    df = base_df[['Date', 'Close']].copy()
    # 计算均线
    df['12EMA'] = df['Close'].ewm(span=12).mean()
    df['144EMA'] = df['Close'].ewm(span=144).mean()
    df['169EMA'] = df['Close'].ewm(span=169).mean()
    df['576EMA'] = df['Close'].ewm(span=576).mean()
    df['676EMA'] = df['Close'].ewm(span=676).mean()

    # 计算斐波那契位置
    df['fib1'] = df['144EMA'] + 0.618 * (df['169EMA'] - df['144EMA'])
    df['fib2'] = df['144EMA'] + 1.618 * (df['169EMA'] - df['144EMA'])
    df['fib3'] = df['144EMA'] + 2.618 * (df['169EMA'] - df['144EMA'])
    df['fib4'] = df['144EMA'] + 4.236 * (df['169EMA'] - df['144EMA'])
    df['fib5'] = df['144EMA'] + 6.854 * (df['169EMA'] - df['144EMA'])

    # 进入隧道区间的信号
    df['in_tunnel'] = np.where((df['Close'] > df['12EMA']) & (df['Close'] < df['144EMA']), True, False)

    # 突破隧道的信号
    df['breakout'] = np.where((df['Close'] > df['144EMA']) & (df['12EMA'] > df['144EMA']), True, False)

    # 突破上轨和下轨的信号
    df['long_signal'] = np.where((df['breakout'] == True) & (df['Close'] > df['fib1']), True, False)
    df['short_signal'] = np.where((df['breakout'] == True) & (df['Close'] < df['fib1']), True, False)

    return df.long_signal


def weighted_daily_return(df_subset):
    total_time_cost = df_subset['time_cost'].sum() or 1
    weighted_return = np.sum((1 + df_subset['profit']) ** (1 / df_subset['time_cost']) * df_subset['time_cost'])
    wd_return = (weighted_return / total_time_cost) - 1
    return total_time_cost, wd_return


# def back_test(kelly_df, prior, initial_fund=1000, breakdown=False):

#     initial_fund = int(tparam.get('initial_fund', initial_fund))
#     max_invest = initial_fund if np.isnan(tparam.get('max_invest')) else int(np.isnan(tparam.get('max_invest')))
#     r = []
#     total_fund = initial_fund
#     breakdown_rows = []
#     future_profits = {}

#     kelly_df.fillna(0, inplace=True)
#     for today, _close, _profit, _time_cost, pct in kelly_df[['Date', 'Close', 'profit', 'time_cost', 'kelly_f']].values:
#         d = pd.to_datetime(today)
#         # fetch profit for Matured date deals.
#         matured_date = [ fd for fd in future_profits.keys() if fd <= d ]
#         today_profit = sum([sum(future_profits.pop(d)) for d in matured_date])

#         total_fund = total_fund + today_profit
#         investable_fund = total_fund
#         # Only invest when we have enough fund.
#         if total_fund >= 10:
#             _max_invest = min(max_invest, total_fund)
#             # Calculate bet amount
#             _invest, _keep = _max_invest * pct, _max_invest * (1-pct)
#             if _invest:
#                 # register future profit
#                 if _time_cost > 0:
#                     future_date = d + pd.DateOffset(days=_time_cost)
#                     if future_date not in future_profits:
#                         future_profits[future_date] = []
#                     future_profit = _invest * (1 + _profit)
#                     future_profits[future_date].append(future_profit)
#                 else:  # Unknown profit yet
#                     future_date = None
#                     future_profit = None
#                     _profit = None
#             else:
#                 # No invest due to no confidence
#                 _invest = 0
#                 future_date = None
#                 future_profit = None
#                 _profit = None
#             total_fund -= _invest
#         else:
#             # No invest due to out of money
#             _invest = 0
#             future_date = None
#             future_profit = None
#             _profit = None

#         future_date = future_date
#         future_profit = future_profit
#         breakdown_rows.append([today, _close, today_profit, investable_fund, _invest,
#                                _profit, future_profit, future_date,
#                                total_fund])

#     # added unclaim Matured profit back if any.
#     if future_profits:
#         profit_date = [ fd for fd in future_profits.keys() ]

#         today_profit = 1
#         d = None
#         for fd in profit_date:
#             d = fd
#             f_profit = sum(future_profits.pop(fd))
#             today_profit += f_profit
#         total_fund = total_fund + today_profit
#         breakdown_rows.append([d, None, today_profit, None, None, None, None, None, total_fund])

#     sample = len(kelly_df)
#     profit_sample = len(kelly_df[kelly_df.profit > 0])
#     profit_mean = (kelly_df.profit > 0).mean()
#     loss_mean = (kelly_df.profit < 0).mean()
#     profit_loss_ratio = profit_mean / loss_mean
#     cost = initial_fund
#     profit = total_fund - cost
#     time_cost, w_daily_return = weighted_daily_return(kelly_df)
#     avg_time_cost = time_cost/sample if sample else 0
#     avg_profit = profit/sample if sample else 0
#     drawdown = kelly_df.profit.min()

#     _annual_return, _sortino_ratio = calc_annual_return_and_sortino_ratio(cost, profit, kelly_df)
#     r.append([profit, cost, profit_loss_ratio, sample, profit_sample, avg_time_cost, avg_profit, drawdown, _annual_return, _sortino_ratio])
#     profit_table = pd.DataFrame(r, columns=['Profit', 'Cost', 'ProfitLossRatio', 'Sample', 'ProfitSample', 'Avg.Timecost', 'Avg.Profit', 'Drawdown', 'Annual.Return', 'SortinoRatio'])
#     bdown_df = pd.DataFrame(breakdown_rows, columns=['Date', 'Close', 'Income', 'CashOnHand', 'Invest', 'FutureProfit', 'FutProfitAmount', 'FuturePaydate', 'Total_fund'])
#     bdown_df = pd.merge(kelly_df[['Date', 'Posterior', 'kelly_f', 'MaxDrawdown', 'MeanProfit']], bdown_df, how='outer', on=['Date'])
#     bdown_df['Profit.kelly_f'] = (bdown_df['FutureProfit'] > 0) & (bdown_df['kelly_f'] > 0)
#     if breakdown:
#         bdown_df.to_csv(f'reports/simulate_transaction.csv', index=False)
#         print(f'Build: reports/simulate_transaction.csv')

#     return profit_table, w_daily_return, bdown_df[['Date', 'Posterior', 'kelly_f', 'Profit.kelly_f', 'MaxDrawdown', 'MeanProfit', 'Invest', 'FutProfitAmount', 'FuturePaydate', 'Income', 'Total_fund']]

def back_test(base_df, initial_fund=1000, breakdown=False):
    max_invest = initial_fund if np.isnan(tparam.get('max_invest')) else int(tparam.get('max_invest'))
    r = []
    total_fund = initial_fund
    breakdown_rows = []
    future_profits = {}

    # is_scratch = 'Total_fund' not in base_df.columns
    # if is_scratch:
    #     base_df = base_df.assign(Invest=np.nan,
    #                         FutProfitAmount=np.nan,
    #                         Income=np.nan,
    #                         Total_fund=np.nan)
    #     idx = base_df.index
    # else:
    #     idx = base_df.loc[np.isnan(base_df.Total_fund)].index
    #     future_payment = base_df[base_df.Matured.notna()]
    #     new_date = base_df.loc[np.isnan(base_df.Total_fund)].Date.values[0]
    #     for _date, _amt, _matured in future_payment[['Date', 'FutProfitAmount', 'Matured']].values:
    #         if _matured < pd.to_datetime(new_date):
    #             # print(f'Skip: {i} < {new_date}')
    #             pass
    #         else:
    #             print('unPNL')
    #             print(f'Today: {base_df.iloc[-1].Date}')
    #             print(f"{_date}, {_amt}, {_matured}")
    #             print(future_payment)


    df = base_df.copy()
    df.fillna(0, inplace=True)

    for today, _close, _profit, _time_cost, _matured_date, pct in df[['Date', 'Close', 'profit', 'time_cost', 'Matured', 'kelly_f']].values:
        d = pd.to_datetime(today)
        # fetch profit for Matured date deals.
        matured_date = [ fd for fd in future_profits.keys() if fd <= d ]
        today_profit = sum([sum(future_profits.pop(d)) for d in matured_date])

        total_fund = total_fund + today_profit
        investable_fund = total_fund
        # Only invest when we have enough fund.
        if total_fund >= 10:
            _max_invest = min(max_invest, total_fund)
            # Calculate bet amount
            _invest, _keep = _max_invest * pct, _max_invest * (1-pct)
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

        breakdown_rows.append([today, _close, today_profit, investable_fund, _invest,
                               _profit, future_profit, future_date,
                               total_fund])

    # added unclaim Matured profit back if any.
    if future_profits:
        profit_date = [ fd for fd in future_profits.keys() ]

        today_profit = 1
        d = np.nan
        for fd in profit_date:
            d = fd
            f_profit = sum(future_profits.pop(fd))
            today_profit += f_profit
        total_fund = total_fund + today_profit
        breakdown_rows.append([d, np.nan, today_profit, np.nan, np.nan, np.nan, np.nan, np.nan, total_fund])

    sample = len(df)
    profit_sample = len(df[df.profit > 0])
    profit_mean = (df.profit > 0).mean()
    loss_mean = (df.profit < 0).mean()
    profit_loss_ratio = profit_mean / loss_mean
    cost = initial_fund
    profit = total_fund - cost
    time_cost, w_daily_return = weighted_daily_return(df)
    avg_time_cost = time_cost/sample if sample else 0
    avg_profit = profit/sample if sample else 0
    drawdown = df.profit.min()

    _annual_return, _sortino_ratio = calc_annual_return_and_sortino_ratio(cost, profit, df, '1min')
    r.append([profit, cost, profit_loss_ratio, sample, profit_sample, avg_time_cost, avg_profit, drawdown, _annual_return, _sortino_ratio])
    profit_table = pd.DataFrame(r, columns=['Profit', 'Cost', 'ProfitLossRatio', 'Sample', 'ProfitSample', 'Avg.Timecost', 'Avg.Profit', 'Drawdown', 'Annual.Return', 'SortinoRatio'])
    bdown_df = pd.DataFrame(breakdown_rows, columns=['Date', 'Close', 'Income', 'CashOnHand', 'Invest', 'FutureProfit', 'FutProfitAmount', 'FuturePaydate', 'Total_fund'])
    bdown_df = pd.merge(df[['Date', 'Posterior', 'kelly_f', 'MaxDrawdown', 'MeanProfit']], bdown_df, how='outer', on=['Date'])
    bdown_df['Profit.kelly_f'] = (bdown_df['FutureProfit'] > 0) & (bdown_df['kelly_f'] > 0)
    if breakdown:
        bdown_df.to_csv(f'reports/simulate_transaction.csv', index=False)
        print(f'Build: reports/simulate_transaction.csv')

    return profit_table, w_daily_return, bdown_df[['Date', 'Posterior', 'kelly_f', 'Profit.kelly_f', 'MaxDrawdown', 'MeanProfit', 'Invest', 'FutProfitAmount', 'FuturePaydate', 'Income', 'Total_fund']]


def calc_confidence_betratio(base_df, signal_func, prior=0.5, debug=False):
    base_df = enrichment_daily_profit(base_df)
    windows = int(tparam.get('bayes_windows', 120))
    _prior = odd(prior)
    _rs = []
    _desc = []

    sell_idx = base_df[np.isnan(base_df['sell'])].index
    sell_windows = (pd.to_datetime(base_df.iloc[-1].Date) - pd.to_datetime(base_df.loc[sell_idx[0]].Date)).days

    is_scratch = 'kelly_f' not in base_df.columns
    if is_scratch:
        base_df = base_df.assign(Prior=np.nan,
                                    Likelihood=np.nan,
                                    Posterior=np.nan,
                                    kelly_f=np.nan,
                                    MaxDrawdown=np.nan,
                                    MeanProfit=np.nan)

    windows = max(windows, sell_windows)
    mv_window = len(base_df) if is_scratch else windows
    df = base_df.iloc[-mv_window:].copy()


    s_signal = signal_func(df)
    # s_signal = s_signal.reset_index(drop=False)

    if is_scratch:
        bayes_idx = df[s_signal].index
    else:
        mask = ~np.isnan(df.Posterior)
        _prior = df.loc[mask, 'Posterior'].iloc[-1]
        _prior = odd(_prior)
        bayes_idx = df[s_signal].loc[np.isnan(df.Posterior)].index

    prior = []
    posterior = []
    like = []
    max_drawdown = []
    mean_profit = []
    kelly = []

    for today in df.loc[bayes_idx].Date:
        s_matured, s_eod, s_eod_yet_matured = pick_dates(df, today, windows)
        sub_df = enrichment_temp_close(df[s_eod_yet_matured], today)
        df.loc[sub_df.index, ['sell', 'time_cost', 'Matured', 'profit']] = sub_df[['sell', 'time_cost', 'Matured', 'profit']]

        # Noted. after using temp close solution, we should calculate all eod trades instead of matured.
        s_profit = s_eod & (df.profit > 0)
        # s_profit = s_matured & (df.profit > 0)

        s_loss = s_eod & ~s_profit
        w_sub_signal = [0.9]
        # for name, s_signal in self._book.get(today):
        #     _signal_posterior, _signal_kelly, exp_profit, debug_list = self.sub_signal(name, today)
        #     w_sub_signal.append(0.9 if name == 'TurtleBuy' else 0.1)
        #     # only take valueable buy signal.
        #     # if _signal_kelly > 0:
        #     #     # print(f'add {today} signal: {name} due to Kelly: {_signal_kelly}')
        #     #     self._df.loc[self._df.Date == today, 'Signal'] = True
        #     self._df.loc[self._df.Date == today, 'Signal'] = True
        #     self._df.loc[self._df.Date == today, 'Source'] = name
        #     self._df.loc[self._df.Date == today, 'SignalW'] = _signal_kelly
        #     self._df.loc[self._df.Date == today, 'SignalDebug'] = str(debug_list)
        #     self._df.loc[self._df.Date == today, 'SignalExpectProfit'] = exp_profit
        #     self._df.loc[self._df.Date == today, 'SignalCnt'] = len(self._book.get(today))

        # ok, start to recalce new signal
        # s_mix_signal = self._df.Signal
        s_mix_signal = s_signal
        _like, debug_list = calc_likelihood(s_profit, s_loss, s_mix_signal)
        _posterior = _prior * _like
        # _posterior = _prior * adjust_like(sum(w_sub_signal), _like)
        # print(f'Pri: {_prior} * Like: {_like} = Post {_posterior}')
        _max_drawdown = abs(-1) if np.isnan(df[s_loss].profit.min()) else abs(df[s_loss].profit.min())
        _mean_profit = 0 if np.isnan(df[s_profit].profit.mean()) else df[s_profit].profit.mean()
        _kelly_f = kelly_formular(pwin=prob_odd(_posterior), loss_margin=_max_drawdown, profit_margin=_mean_profit)

        prior.append(prob_odd(_prior))
        like.append(_like)
        posterior.append(prob_odd(_posterior))
        kelly.append(_kelly_f)
        max_drawdown.append(_max_drawdown)
        mean_profit.append(_mean_profit)

        # update
        _prior = _posterior
        df = base_df.iloc[-mv_window:].copy()

    base_df.loc[bayes_idx, 'Prior'] = prior
    base_df.loc[bayes_idx, 'Likelihood'] = like
    base_df.loc[bayes_idx, 'Posterior'] = posterior
    base_df.loc[bayes_idx, 'MaxDrawdown'] = max_drawdown
    base_df.loc[bayes_idx, 'MeanProfit'] = mean_profit
    base_df.loc[bayes_idx, 'kelly_f'] = kelly
    # _kelly_df = pd.DataFrame(_rs, columns=['Date', 'Close', 'buy', 'sell', 'time_cost', 'profit', 'Matured', 'Prior', 'ttl_name', 'ttl_w', 'ttl_like', 'ttl_profit_count','ttl_loss_count', 'bbl_name', 'bbl_w', 'bbl_like', 'bbl_profit_count','bbl_loss_count', 'Likelihood', 'Posterior', 'kelly_f', 'MaxDrawdown', 'MeanProfit'])
    # if debug:
    #     _kelly_df.to_csv(f'reports/{self._name}_dynamic_kelly.csv', index=False)
    #     print(f'Build: reports/{self._name}_dynamic_kelly.csv')
    #     # self._signal_center.plot_performance()

    # migrate to original history table
    # full_df = pd.merge(df, _bk_df[['Date', 'Posterior', 'kelly_f']], how='left', on=['Date'])
    return base_df

tparam = {
'ATR_sample': 116,
 'atr_loss_margin': 2.00000,
 'bayes_windows': 198,
 'lower_sample': 113,
 'upper_sample': 8,
 'max_invest': 100,
}


# if __name__ == '__main__':
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.float_format', lambda x: '%.5f' % x)
#     pd.set_option('display.width', 300)

#     code = 'SD-USD'
#     df = pd.read_csv(f'data/{code}.csv')
#     df = df.dropna()

#     bkf = BayesKelly(df, name=code)

#     bkf.register_signal('TurtleBuy', s_turtle_buy)
#     kelly_df = bkf.bayes_update(prior=0.5, debug=True)
#     profit_table, w_daily_return, simulate_transaction_df = back_test(kelly_df, prior=0.5, breakdown=True)
#     print(profit_table)
#     full_df = pd.merge(bkf._df, simulate_transaction_df, how='left', on=['Date'])
#     full_df.to_csv(f'reports/{bkf._name}_full_df.csv', index=False)
#     print(f'created: reports/{bkf._name}_full_df.csv')

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    pd.set_option('display.width', 300)

    # code = 'BTC-USD'
    # code = 'SD-USD'
    code = 'btcusdt_backtest'
    df = pd.read_csv(f'data/{code}.csv')
    df = df.dropna()
    size = len(df)

    start_idx = 0
    windows = 200
    # base_df = df.iloc[start_idx:start_idx+windows]
    base_df = df.copy()
    base_df = calc_confidence_betratio(base_df, s_turtle_buy)
    last_row = df.iloc[-1].Date
    # for i in range(size):
    #     new_data = df.iloc[start_idx+windows+i]
    #     base_df = base_df.append(new_data)
    #     base_df = calc_confidence_betratio(base_df, s_turtle_buy)
    #     if last_row == base_df.iloc[-1].Date:
    #         break

    profit_table, w_daily_return, simulate_transaction_df = back_test(base_df, breakdown=False)
    print(profit_table)
    merge_backtest = True
    if merge_backtest:
        print(base_df.tail(10))
        base_df = pd.merge(base_df, simulate_transaction_df[['Date', 'Profit.kelly_f', 'Invest', 'Income', 'Total_fund']], how='left', on=['Date'])
        base_df.to_csv(f'reports/{code}_backtest.csv', index=False)
        print(f'created: reports/{code}_backtest.csv')
    else:
        base_df.to_csv(f'reports/{code}_signal.csv', index=False)
        print(f'created: reports/{code}_signal.csv')

    print(base_df[terminal_view].tail(60))
