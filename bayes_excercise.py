import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import functools
from bayes import conditional, prob


def enricment_turtle(df, upper_sample=20, lower_sample=10, ATR_sample=20):
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


def turtle_sell_strategy(df, upper_sample=20, lower_sample=10, ATR_sample=20):
    df = enricment_turtle(df, upper_sample, lower_sample, ATR_sample)

    sell = []
    time_cost = []
    for i, _v in enumerate(zip(df.buy.values, df.ATR.values)):
        buy, buy_atr = _v
        buy_atr = buy_atr * 2

        sell_point = None
        days = None
        for j, v in enumerate(zip(df.Close.values[i+1:], df.turtle_l.values[i+1:], df.ATR.values[i+1:])):
            _close, _turtle_low, _atr = v
            sell_point, days = (_close, j) if (_atr > buy_atr) or (_close < _turtle_low) else (None, None)
            if sell_point:
                break

        if sell_point:
            sell.append(sell_point)
            time_cost.append(max(days, 0.5))
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
    df.loc[:, 'profit'] = (df.sell / df.buy) - 1
    return df



class Strategy(object):


    def __init__(self, df, func_sell_strategy=None):
        func_sell_strategy = (func_sell_strategy
                                or functools.partial(turtle_sell_strategy,
                                                     upper_sample=14,
                                                     lower_sample=7,
                                                     ATR_sample=14))
        # func_sell_strategy = (func_sell_strategy
        #                         or functools.partial(grid_sell_strategy,
        #                                              stop_loss=1-loss_margin,
        #                                              take_profit=1+profit_margin)
        self.df = enrichment_daily_profit(df, func_sell_strategy)
        self.profit = df.profit > 0
        self.tt_buy = (df.Close > df.turtle_h)


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


    def bayes_table(self):
        r = []
        profit = self.profit
        for name in self.names():
            r.append([name, conditional(self.cond(name), profit), prob(self.cond(name)), prob(profit)])

        rx = pd.DataFrame(r, columns=['Key', 'P(D|profit)', 'P(D)', 'prior'])
        rx['likelihood'] = rx['P(D|profit)']/rx['P(D)']
        rx['posterior'] = prob(profit) * rx['likelihood']
        return rx


    def kelly_table(self, fund):
        rx = self.bayes_table()
        pwin = rx['posterior']
        plose = 1 - pwin    
        loss = loss_margin
        profit = profit_margin
        rx['kelly(f)'] = ((pwin/loss - plose/profit)+20)/40
        rx['Invest'] = fund * rx['kelly(f)']
        # print('Stop loss/Take profit: (%s, %s)' % (loss, profit))
        # print('--' * 30)
        return rx


def split_train_test(df, ratio):
    train_df = df.index < int(len(df) * ratio)
    test_df = ~train_df
    return df[train_df], df[test_df]


def weighted_daily_return(df_subset):
    total_time_cost = df_subset['time_cost'].sum()
    weighted_return = np.sum((1 + df_subset['profit']) ** (1 / df_subset['time_cost']) * df_subset['time_cost'])
    wd_return = (weighted_return / total_time_cost) - 1
    return wd_return


def back_test(test_name, strategy_obj, kelly_df, initial_fund=1000, breakdown=[]):
    s = strategy_obj
    total_profit = 0
    total_cost = 0
    total_sample = 0
    kelly_df.fillna(0, inplace=True)
    funding_distribution = dict(zip(kelly_df.Key, kelly_df['kelly(f)']))
    r = []

    annualized_returns = 0
    w_daily_return = 0
    for name, pct in funding_distribution.items():
        if name == 'profit':
            continue
        if not pct:
            continue

        total_fund = initial_fund
        df_subset = s.dataset(name)

        df_subset = df_subset[df_subset.sell > 0]
        breakdown_rows = []
        future_profits = {}
        total_profit = 0
        for d, _profit, _time_cost in zip(df_subset.Date.values, df_subset.profit.values, df_subset.time_cost.values):

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

                # register future profit
                future_date = d + pd.DateOffset(days=_time_cost)
                if future_date not in future_profits:
                    future_profits[future_date] = []
                future_profit = _invest * (1 + _profit)
                future_profits[future_date].append(future_profit)

            total_fund = _keep
            future_date = future_date or "No Invest"
            future_profit = future_profit or 0
            breakdown_rows.append([d, today_profit, investable_fund, _invest,
                                   _profit, future_profit, future_date,
                                   total_fund])


        # added unclaim Matured profit back if any.
        if future_profits:
            profit_date = [ fd for fd in future_profits.keys() ]

            today_profit = 0
            d = None
            for fd in profit_date:
                d = fd
                f_profit = sum(future_profits.pop(fd))
                today_profit += f_profit
            total_fund = total_fund + today_profit
            breakdown_rows.append([d, today_profit, None, None, None, None, None, total_fund])

        if name in breakdown:
            bdown_df = pd.DataFrame(breakdown_rows, columns=['Date', 'Paid', 'Investable Fund', 'Invest', 'F.Profit', 'F.ProfitAmount', 'F.paydate', 'Total_fund'])
            bdown_df.to_csv(f'breakdown_{name}.csv', index=False)

        sample = len(df_subset)
        cost = initial_fund
        profit = total_fund - cost
        profit_ratio = profit/initial_fund

        w_daily_return = weighted_daily_return(df_subset)
        time_cost = df_subset.time_cost.sum()
        total_profit += profit
        total_cost += cost
        total_sample += sample
        avg_time_cost = time_cost/sample if sample else 0
        avg_profit = profit/sample if sample else 0
        drawdown = df_subset.profit.min()
        upper_sample, lower_sample, ATR_sample = test_name.split('_')[1:]
        r.append([test_name, upper_sample, lower_sample, ATR_sample, profit, cost, profit/cost if cost != 0 else 0, time_cost, sample, avg_time_cost, avg_profit, drawdown])
    profit_table = pd.DataFrame(r, columns=['Cond', 'upper_sample', 'lower_sample', 'ATR_sample', 'Profit', 'Cost', 'ProfitRatio', 'Timecost', 'Sample', 'Avg.Timecost', 'Avg.Profit', 'Drawdown'])



    loss = 1 - loss_margin
    profit = 1 + profit_margin
    # print(f'Stop loss/Take profit: ({loss}, {profit})')
    # print('=' * 100)
    return profit_table, w_daily_return


loss_margin = 0.05
profit_margin = 0.05

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x) 
    pd.set_option('display.width', 300)

    df = pd.read_csv('data/NEAR-USD.csv')
    # df = pd.read_csv('data/BTC-USD.csv')
    # df = pd.read_csv('data/7113.KL.csv')

    df = df.dropna()
    train, test = split_train_test(df, 0.2)
    print(f"Train range: {train.Date.values[0]} - {train.Date.values[-1]}")

    # Best params: {'ATR_sample': 60.97992422245601, 'lower_sample': 6.657677183757127, 'upper_sample': 34.10890856894943}
    upper_sample = 34
    lower_sample = 7
    ATR_sample = 61

    turtle_sell = functools.partial(turtle_sell_strategy, upper_sample=upper_sample,
                                    lower_sample=lower_sample, ATR_sample=ATR_sample)
    s = Strategy(train, turtle_sell)
    kelly_df = s.kelly_table(fund=100)

    test_strategy = Strategy(test, turtle_sell)
    test_name = f'TTS_{upper_sample}_{lower_sample}_{ATR_sample}'
    # back test
    profit_table, w_daily_return = back_test(test_name, test_strategy, kelly_df, breakdown=['tt_buy'])
    performance_table = pd.DataFrame(profit_table, columns=['Cond', 'upper_sample', 'lower_sample', 'ATR_sample', 'Profit', 'Cost', 'ProfitRatio', 'Timecost', 'Sample', 'Avg.Timecost', 'Avg.Profit', 'Drawdown'])
    print(f"{upper_sample}|{lower_sample}|{ATR_sample}={w_daily_return}")
    # print('-'*100)
    # print(performance_table)
    performance_table.to_csv('performance.csv', index=False)
    print(performance_table)






