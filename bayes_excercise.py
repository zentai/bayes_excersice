import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import functools
from bayes import conditional, prob


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
    df['buy'] = df.Open.shift(-1)
    sell, time_cost = func_sell_strategy(df)
    df['sell'] = sell
    df['time_cost'] = time_cost
    df['profit'] = (df.sell / df.buy) - 1
    return df


class Strategy(object):


    def __init__(self, df):
        func_sell_strategy = functools.partial(grid_sell_strategy, stop_loss=1-loss_margin, take_profit=1+profit_margin)
        self.df = enrichment_daily_profit(df, func_sell_strategy)
        self.profit = df.profit > 0


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
        df[name+'_entry'] = np.where(self.cond(name), df.Close.max(), 0)
        df.plot(x='Date', y=['Close', name+'_entry'])
        plt.show()


    def plot_performance(self, name):
        if not hasattr(self, name):
            return None
        df = self.df.copy()
        close_mean = df.Close.max()
        norm = df[name] / df[name].max()
        df[name+'_entry'] = norm * close_mean
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
            r.append([name, conditional(s.cond(name), profit), prob(s.cond(name)), prob(profit)])

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
        rx['kelly(f)'] = pwin/loss - plose/profit
        rx['Invest'] = fund * rx['kelly(f)']
        print('Stop loss/Take profit: (%s, %s)' % (loss, profit))
        print('--' * 30)
        return rx


def split_train_test(df, ratio):
    train_df = df.index < int(len(df) * ratio)
    test_df = ~train_df
    return df[train_df], df[test_df]

loss_margin = 0.05
profit_margin = 0.05

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x) 
    pd.set_option('display.width', 300)
    df = pd.read_csv('data/BTC-USD.csv')
    df = df.dropna()
    train, test = split_train_test(df, 0.8)
    s = Strategy(train)
    print(s.df)
    print(s.kelly_table(fund=100))





