import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import functools
from bayes import conditional, prob


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
    df['buy'] = df.Open.shift(-1)
    sell, time_cost = func_sell_strategy(df)
    df['sell'] = sell
    df['time_cost'] = time_cost
    df['profit'] = (df.sell / df.buy) - 1
    return df


def enrichment_flags(df):
    pass


def enricment_turtle(df, upper_sample=20, lower_sample=10, ATR_sample=20):
    tt_df = pd.DataFrame()
    tt_df['Date'] = df.Date
    tt_df['Close'] = df.Close
    tt_df['turtle_h'] = df.Close.shift(1).rolling(upper_sample).max()
    tt_df['turtle_l'] = df.Close.shift(1).rolling(lower_sample).min()
    tt_df['h-l'] = df.High - df.Low
    tt_df['c-h'] = (df.Close.shift(1)-df.High).abs()
    tt_df['c-l'] = (df.Close.shift(1)-df.Low).abs()
    tt_df['TR'] = tt_df[['h-l', 'c-h', 'c-l']].max(axis=1)
    tt_df['ATR'] = (tt_df.TR.rolling(ATR_sample).sum()/ATR_sample)
    df['turtle_h'] = tt_df['turtle_h']
    df['turtle_l'] = tt_df['turtle_l']
    df['ATR'] = tt_df['ATR']
    return df

class Strategy(object):


    def __init__(self, df):
        func_sell_strategy = functools.partial(turtle_sell_strategy, upper_sample=30, lower_sample=10, ATR_sample=30)
        # func_sell_strategy = functools.partial(grid_sell_strategy, stop_loss=1-loss_margin, take_profit=1+profit_margin)
        self.df = enrichment_daily_profit(df, func_sell_strategy)
        self.df = enrichment_flags(self.df)
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
    train, test = split_train_test(df, 0.2)
    s = Strategy(train)
    print(s.df)
    print(s.kelly_table(fund=100))





