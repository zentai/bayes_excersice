import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import functools
from bayes import conditional, prob, odd, prob_odd


def calc_annual_return_and_sortino_ratio(cost, profit, df):
    _yield_curve_1yr = 0.0419
    _start_date = df.iloc[0].Date
    _end_date = df.iloc[-1].Date
    _trade_count = len(df)
    _trade_days = (pd.to_datetime(_end_date) - pd.to_datetime(_start_date)).days

    _annual_trade_count = (_trade_count / _trade_days) * 365
    _downside_risk_stdv = df[df.profit < _yield_curve_1yr].profit.std(ddof=1)
    _annual_downside_risk_stdv = _downside_risk_stdv * np.sqrt(_annual_trade_count)

    t = _trade_days / 365
    _annual_return = (profit / cost) ** (1/t) - 1 if (profit / cost > 0) else 0
    _sortino_ratio = (_annual_return - _yield_curve_1yr) / _annual_downside_risk_stdv
    return _annual_return, _sortino_ratio


def enrichment_daily_profit(df):
    _loss_margin = tparam.get('atr_loss_margin', 1.5)
    df = turtle_trading(df)
    df.loc[:, 'buy'] = df.Open.shift(-1)
    stop_profits = []
    sell = []
    time_cost = []
    for i, _v in enumerate(zip(df.buy.values, df.ATR.values)):
        buy, buy_atr = _v

        sell_point = None
        days = None
        _pre_close = buy
        stop_profit = _pre_close - buy_atr * _loss_margin
        for j, v in enumerate(zip(df.Close.values[i+1:], df.turtle_l.values[i+1:], df.ATR.values[i+1:])):
            _close, _turtle_low, _atr = v
            sell_point, days = (_close, j) if (_close < _pre_close - _atr * _loss_margin) or (_close < _turtle_low) else (None, None)
            if sell_point:
                break
            _pre_close = _close
        if sell_point:
            sell.append(sell_point)
            stop_profits.append(stop_profit)
            time_cost.append(days+1)
        else:
            sell.append(None)
            stop_profits.append(stop_profit)
            time_cost.append(None)
    df.loc[:, 'sell'] = sell
    df.loc[:, 'time_cost'] = time_cost
    df.loc[:, 'Matured'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['time_cost'], 'd')
    df.loc[:, 'Stop_profit'] = stop_profits
    df.loc[:, 'profit'] = (df.sell / df.buy) - 1
    return df


def enrichment_temp_close(df, today):
    close = df[df.Date==today].Close.values[-1]
    df.iloc[-1, df.columns.get_loc('buy')] = close      # we always take next day open as buy, but since we remove T+1, so we consider no profit at the last day
    df.loc[:, 'sell'] = close
    df.loc[:, 'time_cost'] = (pd.to_datetime(today) - pd.to_datetime(df['Date'])).dt.days
    df.loc[:, 'Matured'] = pd.to_datetime(today)
    df.loc[:, 'profit'] = (df.sell / df.buy) - 1
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
    df = df[['Date', 'Matured']].copy()
    df['Date'] = pd.to_datetime(df.Date)
    start_day = df.Date >= (pd.to_datetime(today) - pd.to_timedelta(windows, 'd'))
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

        _kelly_df = pd.DataFrame(_rs, columns=['Date', 'Close', 'buy', 'sell', 'time_cost', 'profit', 'Matured', 'Prior', 'ttl_name', 'ttl_w', 'ttl_like', 'ttl_profit_count','ttl_loss_count', 'bbl_name', 'bbl_w', 'bbl_like', 'bbl_profit_count','bbl_loss_count', 'Likelihood', 'Posterior', 'kelly(f)', 'Max.Drawdown', 'Mean.Profit'])
        if debug:
            _kelly_df.to_csv(f'reports/{self._name}_dynamic_kelly.csv', index=False)
            print(f'Build: reports/{self._name}_dynamic_kelly.csv')
            # self._signal_center.plot_performance()

        # migrate to original history table
        # full_df = pd.merge(df, _bk_df[['Date', 'Posterior', 'kelly(f)']], how='left', on=['Date'])
        return _kelly_df


def turtle_trading(base_df):
    upper_sample = int(tparam.get('upper_sample', 20) )
    lower_sample = int(tparam.get('lower_sample', 10) )
    ATR_sample = int(tparam.get('ATR_sample', 20) )

    df = base_df[['Date', 'Open', 'Close', 'High', 'Low']].copy()
    df.loc[:, 'turtle_h'] = df.Close.shift(1).rolling(upper_sample).max()
    df.loc[:, 'turtle_l'] = df.Close.shift(1).rolling(lower_sample).min()
    df.loc[:, 'h-l'] = df.High - df.Low
    df.loc[:, 'c-h'] = (df.Close.shift(1)-df.High).abs()
    df.loc[:, 'c-l'] = (df.Close.shift(1)-df.Low).abs()
    df.loc[:, 'TR'] = df[['h-l', 'c-h', 'c-l']].max(axis=1)
    df.loc[:, 'ATR'] = (df.TR.rolling(ATR_sample).sum()/ATR_sample)
    return df


def s_turtle_buy(base_df):
    df = turtle_trading(base_df)
    return df.Close > df.turtle_h


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


def back_test(kelly_df, prior, initial_fund=1000, breakdown=False):

    initial_fund = int(tparam.get('initial_fund', initial_fund))
    max_invest = initial_fund if np.isnan(tparam.get('max_invest')) else int(np.isnan(tparam.get('max_invest')))
    r = []
    total_fund = initial_fund
    breakdown_rows = []
    future_profits = {}

    kelly_df.fillna(0, inplace=True)
    for today, _close, _profit, _time_cost, pct in kelly_df[['Date', 'Close', 'profit', 'time_cost', 'kelly(f)']].values:
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
                    future_date = d + pd.DateOffset(days=_time_cost)
                    if future_date not in future_profits:
                        future_profits[future_date] = []
                    future_profit = _invest * (1 + _profit)
                    future_profits[future_date].append(future_profit)
                else:  # Unknown profit yet
                    future_date = None
                    future_profit = None
                    _profit = None
            else:
                # No invest due to no confidence
                _invest = 0
                future_date = None
                future_profit = None
                _profit = None
            total_fund -= _invest
        else:
            # No invest due to out of money
            _invest = 0
            future_date = None
            future_profit = None
            _profit = None

        future_date = future_date
        future_profit = future_profit
        breakdown_rows.append([today, _close, today_profit, investable_fund, _invest,
                               _profit, future_profit, future_date,
                               total_fund])

    # added unclaim Matured profit back if any.
    if future_profits:
        profit_date = [ fd for fd in future_profits.keys() ]

        today_profit = 1
        d = None
        for fd in profit_date:
            d = fd
            f_profit = sum(future_profits.pop(fd))
            today_profit += f_profit
        total_fund = total_fund + today_profit
        breakdown_rows.append([d, None, today_profit, None, None, None, None, None, total_fund])

    sample = len(kelly_df)
    profit_sample = len(kelly_df[kelly_df.profit > 0])
    profit_mean = (kelly_df.profit > 0).mean()
    loss_mean = (kelly_df.profit < 0).mean()
    profit_loss_ratio = profit_mean / loss_mean
    cost = initial_fund
    profit = total_fund - cost
    time_cost, w_daily_return = weighted_daily_return(kelly_df)
    avg_time_cost = time_cost/sample if sample else 0
    avg_profit = profit/sample if sample else 0
    drawdown = kelly_df.profit.min()

    _annual_return, _sortino_ratio = calc_annual_return_and_sortino_ratio(cost, profit, kelly_df)
    r.append([profit, cost, profit_loss_ratio, sample, profit_sample, avg_time_cost, avg_profit, drawdown, _annual_return, _sortino_ratio])
    profit_table = pd.DataFrame(r, columns=['Profit', 'Cost', 'ProfitLossRatio', 'Sample', 'ProfitSample', 'Avg.Timecost', 'Avg.Profit', 'Drawdown', 'Annual.Return', 'SortinoRatio'])
    bdown_df = pd.DataFrame(breakdown_rows, columns=['Date', 'Close', 'Income', 'Investable Fund', 'Invest', 'F.Profit', 'F.ProfitAmount', 'F.paydate', 'Total_fund'])
    bdown_df = pd.merge(kelly_df[['Date', 'Posterior', 'kelly(f)', 'Max.Drawdown', 'Mean.Profit']], bdown_df, how='outer', on=['Date'])
    bdown_df['Profit.kelly(f)'] = (bdown_df['F.Profit'] > 0) & (bdown_df['kelly(f)'] > 0)
    if breakdown:
        bdown_df.to_csv(f'reports/simulate_transaction.csv', index=False)
        print(f'Build: reports/simulate_transaction.csv')

    return profit_table, w_daily_return, bdown_df[['Date', 'Posterior', 'kelly(f)', 'Profit.kelly(f)', 'Max.Drawdown', 'Mean.Profit', 'Invest', 'F.ProfitAmount', 'F.paydate', 'Income', 'Total_fund']]


tparam = {
'ATR_sample': 116,
 'atr_loss_margin': 2.00000,
 'bayes_windows': 198,
 'lower_sample': 113,
 'upper_sample': 8,
    'max_invest': np.nan,
}


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x) 
    pd.set_option('display.width', 300)

    code = 'SD-USD'
    df = pd.read_csv(f'data/{code}.csv')
    df = df.dropna()

    bkf = BayesKelly(df, name=code)

    bkf.register_signal('TurtleBuy', s_turtle_buy)
    kelly_df = bkf.bayes_update(prior=0.5, debug=True)
    profit_table, w_daily_return, simulate_transaction_df = back_test(kelly_df, prior=0.5, breakdown=True)
    print(profit_table)
    full_df = pd.merge(bkf._df, simulate_transaction_df, how='left', on=['Date'])
    full_df.to_csv(f'reports/{bkf._name}_full_df.csv', index=False)
    print(f'created: reports/{bkf._name}_full_df.csv')
