import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import functools
from bayes import conditional, prob


def turtle_sell_strategy(df):
    df = turtle_trading(df)

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


def grid_sell_strategy(df):
    stop_loss = tparam.get('stop_loss', 0.5)
    take_profit = tparam.get('take_profit', 0.5)
    
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


class BayesKelly:

    def __init__(self, df):
        self._df = enrichment_daily_profit(df)
        self._book = {}
        self._signals = {}
        self._latest_prior = 0.5

    def register_signal(self, name, preproccess_func):
        s_signal = preproccess_func(self._df)
        self._signals[name] = s_signal
        dates = self._df[s_signal].Date.values
        for d in dates:
            if d not in self._book:
                self._book[d] = []
            self._book[d].append((name, s_signal))

    # TODO: please complete me in new code.
    def bayes_update(self, prior=0.5, windows=1000):
        df = self._df
        _prior = odd(prior)

        for today in sorted(self._book.keys()):
            pass
        return None


def turtle_trading(base_df):
    upper_sample = tparam.get('upper_sample', 20) 
    lower_sample = tparam.get('lower_sample', 10) 
    ATR_sample = tparam.get('ATR_sample', 20) 

    df = base_df[['Date', 'Close', 'High', 'Low']].copy()
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


tparam = {
    # Turtle Trading param
    'upper_sample': 34,
    'lower_sample': 7,
    'ATR_sample': 61,

    # Bollinger band
    'bbh_window': 5,
    'bbl_window': 5,
}


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x) 
    pd.set_option('display.width', 300)

    df = pd.read_csv('data/NEAR-USD.csv')
    # df = pd.read_csv('data/BTC-USD.csv')
    # df = pd.read_csv('data/7113.KL.csv')

    df = df.dropna()
    bkf = BayesKelly(df)
    bkf.register_signal('TurtleBuy', s_turtle_buy)
    bkf.register_signal('BollingerBuy', s_bollinger_band)
    kelly_df = bkf.bayes_update(prior=0.5, windows=720)
    back_test(kelly_df)