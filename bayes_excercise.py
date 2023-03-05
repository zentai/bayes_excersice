import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import functools
from bayes import conditional, prob, odd, prob_odd
from collections import deque


def enrich_mean_reversion(df, mv_a, mv_b, mv_zscor):
    ma_1 = df.Close.rolling(window=mv_a).mean()
    ma_2 = df.Close.rolling(window=mv_b).mean()
    ma_mix = ma_1 - ma_2

    ma = ma_mix.rolling(window=mv_zscor).mean()
    std = df.Close.rolling(window=mv_zscor).std()
    df['ma_1'] = ma_1
    df['ma_2'] = ma_2
    df['ma_mix'] = ma_mix
    df['mr'] = (ma_mix - ma) / std
    return df


def build_bollinger_band(df, mv):
    bollinger_config = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

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

    boll_statis(df, 'BBL', bollinger_config, mv)
    boll_statis(df, 'BBH', bollinger_config, mv)
    return df


def calc_likelihood(s_profit, s_loss, s_signal):

    s_profit = s_profit & s_signal
    s_loss = s_loss & s_signal

    # _proft_count = (s_profit & s_signal).sum()
    # _loss_count = (s_loss & s_signal).sum()
    # _signal_count = len(s_signal)
    # w = (_proft_count+_loss_count)/_signal_count

    _p_prof_v_signal = conditional(s_profit, s_signal)
    _p_loss_v_signal = conditional(s_loss, s_signal)


    if not (_p_prof_v_signal or _p_loss_v_signal):
        # print(f'0/0 = {_p_prof_v_signal} / {_p_loss_v_signal}')
        return 1, _p_prof_v_signal, _p_loss_v_signal
    if _p_prof_v_signal and not _p_loss_v_signal:    # x/0
        # print(f'{_p_prof_v_signal+0.1} / 1 = {(_p_prof_v_signal+0.1) / 1}')
        return (_p_prof_v_signal+0.1) / 1, _p_prof_v_signal, _p_loss_v_signal

    if not _p_prof_v_signal and _p_loss_v_signal:    # 0/x
        # print(f'1 / {_p_loss_v_signal+0.1} = {1/(_p_loss_v_signal+0.1)}')
        return 1 / (_p_loss_v_signal+0.1), _p_prof_v_signal, _p_loss_v_signal

    return _p_prof_v_signal / _p_loss_v_signal, _p_prof_v_signal, _p_loss_v_signal


class BuySignalCenter:
    def __init__(self, df):
        self._df = df.copy()
        self._book = {}
        self._signals = {}

    def register_signal(self, name, s_signal):
        self._signals[name] = s_signal
        for d in self._df[s_signal].Date.values:
            if d not in self._book:
                self._book[d] = []
            self._book[d].append((name, s_signal))


    def signals(self, date):
        return self._book.get(date, [])


    def plot_performance(self):
        self._df['Date'] = pd.to_datetime(self._df.Date)
        plt.plot(self._df.Date, self._df.Close)
        for name, s_signal in self._signals.items():
            buy_dates = self._df.loc[s_signal, 'Date']
            sell_date = self._df.loc[s_signal, 'Matured']
            plt.scatter(buy_dates, self._df.loc[s_signal, 'buy'], marker='^', s=100, color='green', label='B')
            plt.scatter(sell_date, self._df.loc[s_signal, 'sell'], marker='v', s=100, color='red', label='S')
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.show()


def kelly_formular(pwin, loss_margin, profit_margin):
    if loss_margin and profit_margin:
        _pwin = pwin / loss_margin
        _ploss = (1 - pwin) / profit_margin
        return (_pwin - _ploss) / _pwin if _pwin > _ploss else 0
    return 0


def enrich_turtle(df, upper_sample=20, lower_sample=10, ATR_sample=20):
    # print(upper_sample, lower_sample, ATR_sample)
    tt_df = pd.DataFrame()
    tt_df.loc[:, 'Date'] = df.Date
    tt_df.loc[:, 'Close'] = df.Close
    tt_df.loc[:, 'turtle_h'] = df.Close.shift(1).rolling(upper_sample).max()
    tt_df.loc[:, 'turtle_l'] = df.Close.shift(1).rolling(lower_sample).min()
    tt_df.loc[:, 'h-l'] = df.High - df.Low
    tt_df.loc[:, 'c-h'] = (df.Close.shift(1)-df.High).abs()
    tt_df.loc[:, 'c-l'] = (df.Close.shift(1)-df.Low).abs()
    tt_df.loc[:, 'TR'] = tt_df[['h-l', 'c-h', 'c-l']].max(axis=1)
    tt_df.loc[:, 'ATR'] = (tt_df.TR.rolling(ATR_sample).sum()/ATR_sample)
    df.loc[:, 'turtle_h'] = tt_df['turtle_h']
    df.loc[:, 'turtle_l'] = tt_df['turtle_l']
    df.loc[:, 'ATR'] = tt_df['ATR']
    return df


def turtle_sell_strategy(df, upper_sample=20, lower_sample=10, ATR_sample=20, stop_loss=0.95):
    df = enrich_turtle(df, upper_sample, lower_sample, ATR_sample)

    sell = []
    time_cost = []
    for i, _v in enumerate(zip(df.buy.values, df.ATR.values)):
        buy, buy_atr = _v
        buy_atr = buy_atr * 2
        _stop_loss = buy * stop_loss

        sell_point = None
        days = None
        for j, v in enumerate(zip(df.Close.values[i+1:], df.turtle_l.values[i+1:], df.ATR.values[i+1:])):
            _close, _turtle_low, _atr = v
            sell_point, days = (_close, j) if (_atr > buy_atr) or (_close < _turtle_low) or (_close <= _stop_loss)else (None, None)
            if sell_point:
                break

        if sell_point:
            sell.append(sell_point)
            time_cost.append(days+1)
        else:
            sell.append(None)
            time_cost.append(None)
    return sell, time_cost


def grid_sell_strategy(df, stop_loss, take_profit):
    sell = []
    time_cost = []
    for i, buy in enumerate(df.buy.values):
        _stop_loss = buy * stop_loss
        _take_profit = buy * take_profit

        sell_point = None
        days = None
        for j, v in enumerate(zip(df.Low.values[i+1:], df.High.values[i+1:])):
            _low, _high = v
            sell_point, days = (_stop_loss, j) if _low < _stop_loss else (None, None)
            sell_point, days = (_take_profit, j) if _high > _take_profit else (None, None)

            if sell_point:
                break

        if sell_point:
            sell.append(sell_point)
            time_cost.append(max(days, 0.5))
        else:
            sell.append(None)
            time_cost.append(None)
    return sell, time_cost


def enrichment_daily_profit(df, func_sell_strategy):
    df.loc[:, 'buy'] = df.Open.shift(-1)
    sell, time_cost = func_sell_strategy(df)
    df.loc[:, 'sell'] = sell
    df.loc[:, 'time_cost'] = time_cost
    df.loc[:, 'Matured'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.time_cost, 'd')
    df.loc[:, 'profit'] = (df.sell / df.buy) - 1

    # vol_rank = df['Volume'].rolling(window=vol_level).apply(lambda x: list(pd.Series(x).rank(pct=False))[-1], raw=False)
    # df.loc[:, 'vol_rank'] = vol_rank
    return df


class Strategy(object):


    def __init__(self, df, mv, mr, func_sell_strategy=None):
        func_sell_strategy = (func_sell_strategy
                                or functools.partial(turtle_sell_strategy,
                                                     upper_sample=34,
                                                     lower_sample=6,
                                                     ATR_sample=60))
        # func_sell_strategy = (func_sell_strategy
        #                         or functools.partial(grid_sell_strategy,
        #                                              stop_loss=1-loss_margin,
        #                                              take_profit=1+profit_margin)
        self.df = enrichment_daily_profit(df, func_sell_strategy)
        self.df = build_bollinger_band(self.df, mv)
        self.df = enrich_mean_reversion(self.df, *mr)

        # create SignalCenter to backtest
        self._signal_center = BuySignalCenter(self.df)


    def register_signal(self, name, s_signal):
        self._signal_center.register_signal(name, s_signal)


    def dataset(self, name):
        return self.df[self.cond(name) if hasattr(self, name) else pd.DataFrame([], columns=self.df.columns)]


    def names(self):
        names = list(vars(self).keys())
        names.remove('df')
        return names


    def cond(self, name):
        if not name:
            return pd.Series()
        return self.__getattribute__(name) if hasattr(self, name) else pd.Series()


    def plot_entry(self, name):
        if not hasattr(self, name):
            return None
        df = self.df.copy()
        df.loc[:, name+'_entry'] = np.where(self.cond(name), df.Close.max(), 0)
        df.plot(x='Date', y=['Close', name+'_entry'])
        plt.show()


    def plot_performance(self, name):
        if not hasattr(self, name):
            return None
        df = self.df.copy()
        close_mean = df.Close.max()
        norm = df[name] / df[name].max()
        df.loc[:, name+'_entry'] = norm * close_mean
        df.plot(x='Date', y=['Close', name+'_entry'])
        plt.show()


    def dump(self, name=''):
        cond = self.cond(name)
        df_sub = self.df[cond] if cond else self.df
        df_sub.to_csv('df_%s.csv'%name, index=False)


    def kelly_table(self, prior=0.5, rolling=180, debug_winrate=False):
        df = self.df

        _prior = odd(prior)

        _rs = []
        _desc = []
        for today in sorted(self._signal_center._book.keys()):
            _i = max(df.loc[df.Date==today].index)

            _start_day = df.Date > df.iloc[max(_i-rolling, 0)].Date


            _fundamental = list(df.loc[_i, ['Close', 'buy', 'sell', 'time_cost', 'profit', 'Matured']].values)

            s_matured = (df.Matured <= today) & _start_day
            s_eod = (df.Date <= today) & _start_day
            s_profit = s_matured & (df.profit > 0)
            s_loss = s_eod & ~s_profit

            _likelihood = 1
            _max_drawdown = []
            _mean_profit = []
            a = [''] * 12
            # mix strategys likelihood
            for name, s_signal in self._signal_center.signals(today):
                _like, _p_prof_v_signal, _p_loss_v_signal = calc_likelihood(s_profit, s_loss, s_signal & s_eod)

                # please consider migrate into calc_likelihood
                _proft_count = (s_profit & s_signal & s_eod).sum()
                _loss_count = (s_loss & s_signal & s_eod).sum()
                _signal_count = s_eod.sum()
                w = (_proft_count+_loss_count)/30
                print(df[s_signal])
                _estimate_profit = np.nanmin([df[s_profit & s_signal].profit.mean(), df[s_profit].profit.mean()])
                _mean_profit.append(_estimate_profit)

                _estimate_loss = _estimate_profit = np.nanmin([df[s_loss & s_signal].profit.min(), df[s_loss].profit.min()])
                _max_drawdown.append(_estimate_loss)
                _desc.append([name, _like, _p_prof_v_signal, _p_loss_v_signal])

                def adjust_like(w, _like):
                    # adjust trust level when data point not engouh
                    if w >= 1:
                        pass
                    elif w < 1 and _like > 1:
                        _like = (_like - 1) * w + 1
                    elif w < 1 and _like < 1:
                        _like = 1-((1 - _like) * w)
                    elif w < 1 and _like == 1:
                        pass
                    return _like

                if name == 'Turtle Trading':
                    a[0] = 'TTL'
                    a[1] = f'{_p_prof_v_signal} ({(s_profit & s_signal & s_eod).sum()})'
                    a[2] = f'{_p_loss_v_signal} ({(s_loss & s_signal & s_eod).sum()})'
                    a[3] = w
                    a[4] = _like
                    a[5] = adjust_like(w, _like)
                elif name == 'MR':
                    print(rolling, today)
                    print(df[s_signal & s_eod])
                    a[6] = 'MR'
                    a[7] = f'{_p_prof_v_signal} ({(s_profit & s_signal & s_eod).sum()})'
                    a[8] = f'{_p_loss_v_signal} ({(s_loss & s_signal & s_eod).sum()})'
                    a[9] = w
                    a[10] = _like
                    a[11] = adjust_like(w, _like)


                # print(f'[likelihood update] Like: {_likelihood} * {_like} -> {_likelihood * _like}  [{_p_prof_v_signal:.2f}/{_p_loss_v_signal:.2f}]')



                _likelihood = _likelihood * adjust_like(w, _like)

            # print(f'[_posterior update] Posterior: {_prior} * {_likelihood} = {_prior * _likelihood}')
            _posterior = _prior * _likelihood

            _max_drawdown = abs(np.min(_max_drawdown)) if _max_drawdown else 1
            _mean_profit = np.mean(_mean_profit) if _mean_profit else 1

            _kelly_f = kelly_formular(pwin=prob_odd(_posterior), loss_margin=_max_drawdown, profit_margin=_mean_profit)
            # print(f'[{today}] kelly(pwin={prob_odd(_posterior)}, loss={_max_drawdown}, porifit={_mean_profit}) = {_kelly_f}')

            # log
            _rs.append([today] + _fundamental + [prob_odd(_prior), _likelihood, prob_odd(_posterior), _kelly_f] + a)

            # update
            _prior = _posterior

        _kelly_df = pd.DataFrame(_rs, columns=['Date', 'Close', 'buy', 'sell', 'time_cost', 'profit', 'Matured', 'Prior', 'Likelihood', 'Posterior', 'kelly(f)', 'ttl_name', 'ttl_like', 'ttl_p_prof_v_signal', 'ttl_p_loss_v_signal','lit_name', 'lit_like', 'lit_p_prof_v_signal', 'lit_p_loss_v_signal','hit_name', 'hit_like', 'hit_p_prof_v_signal', 'hit_p_loss_v_signal'])
        if debug_winrate:
            pd.DataFrame(_desc, columns=['s_signal', 'likelihood', 'win', 'loss']).to_csv('win_rate.csv', index=False)
            _kelly_df.to_csv('dynamic_kelly.csv', index=False)
            print('Build: dynamic_kelly.csv')
            print('Build: win_rate.csv')
            # self._signal_center.plot_performance()
        return _kelly_df


def split_train_test(df, ratio):
    train_df = df.index < int(len(df) * ratio)
    test_df = ~train_df
    return df[train_df].reset_index(drop=True), df[test_df].reset_index(drop=True)


def weighted_daily_return(df_subset):
    total_time_cost = df_subset['time_cost'].sum() or 1
    weighted_return = np.sum((1 + df_subset['profit']) ** (1 / df_subset['time_cost']) * df_subset['time_cost'])
    wd_return = (weighted_return / total_time_cost) - 1
    return total_time_cost, wd_return


def back_test(test_name, strategy_obj, prior, rolling=180, initial_fund=1000, breakdown=False):
    s = strategy_obj
    total_cost = 0
    total_sample = 0
    total_profit = 0

    kelly_df = s.kelly_table(prior=prior, rolling=rolling, debug_winrate=breakdown)
    kelly_df.fillna(0, inplace=True)

    r = []

    annualized_returns = 0
    w_daily_return = 0

    total_fund = initial_fund
    breakdown_rows = []
    future_profits = {}

    for d, _close, _profit, _time_cost, pct in kelly_df[['Date', 'Close', 'profit', 'time_cost', 'kelly(f)']].values:
        d = pd.to_datetime(d)
        # fetch profit for Matured date deals.
        matured_date = [ fd for fd in future_profits.keys() if fd <= d ]
        today_profit = sum([sum(future_profits.pop(d)) for d in matured_date])

        total_fund = total_fund + today_profit
        investable_fund = total_fund
        # Only invest when we have enough fund.
        if total_fund >= 10:
            # Calculate bet amount
            _invest, _keep = total_fund * pct, total_fund * (1-pct)
            if _invest:
                # register future profit
                future_date = d + pd.DateOffset(days=_time_cost)
                if future_date not in future_profits:
                    future_profits[future_date] = []
                future_profit = _invest * (1 + _profit)
                future_profits[future_date].append(future_profit)
            else:
                # No invest due to no confidence
                _invest = 0
                future_date = None
                future_profit = None
                _profit = None
        else:
            # No invest due to out of money
            _invest = 0
            future_date = None
            future_profit = None
            _profit = None

        total_fund = _keep
        future_date = future_date
        future_profit = future_profit
        breakdown_rows.append([d, _close, today_profit, investable_fund, _invest,
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

    if breakdown:
        bdown_df = pd.DataFrame(breakdown_rows, columns=['Date', 'Close', 'Paid', 'Investable Fund', 'Invest', 'F.Profit', 'F.ProfitAmount', 'F.paydate', 'Total_fund'])
        bdown_df.to_csv(f'simulate_transaction.csv', index=False)
        print(f'Build: simulate_transaction.csv')

    sample = len(kelly_df)
    profit_sample = len(kelly_df[kelly_df.profit > 0])
    cost = initial_fund
    profit = total_fund - cost
    profit_ratio = profit/initial_fund
    time_cost, w_daily_return = weighted_daily_return(kelly_df)
    avg_time_cost = time_cost/sample if sample else 0
    avg_profit = profit/sample if sample else 0
    drawdown = kelly_df.profit.min()
    upper_sample, lower_sample, ATR_sample = test_name.split('_')[1:]
    r.append([test_name, upper_sample, lower_sample, ATR_sample, profit, cost, profit/cost if cost != 0 else 0, time_cost, sample, profit_sample, avg_time_cost, avg_profit, drawdown])
    profit_table = pd.DataFrame(r, columns=['Cond', 'upper_sample', 'lower_sample', 'ATR_sample', 'Profit', 'Cost', 'ProfitRatio', 'Timecost', 'Sample', 'ProfitSample', 'Avg.Timecost', 'Avg.Profit', 'Drawdown'])



    loss = 1 - loss_margin
    profit = 1 + profit_margin
    # print(f'Stop loss/Take profit: ({loss}, {profit})')
    # print('=' * 100)
    return profit_table, w_daily_return

loss_margin = 0.05
profit_margin = 0.05
