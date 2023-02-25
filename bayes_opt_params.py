from bayes_opt import BayesianOptimization
from sklearn.model_selection import ParameterGrid
import functools
from bayes_excercise import *
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# 定義目標函數
# def target_function(upper_sample, lower_sample, ATR_sample, stop_loss, rolling, debug=False):
# def target_function(upper_sample=5, lower_sample=5, ATR_sample=5, stop_loss=5, rolling=5, mv_bbh=5, BBHLevel=1, vol_level=1, debug=False):
def target_function(**kwargs):
    # 在這裡使用參數進行回測，計算績效分數
    upper_sample = int(kwargs.get('upper_sample', 20))
    lower_sample = int(kwargs.get('lower_sample', 20))
    ATR_sample = int(kwargs.get('ATR_sample', 20))
    rolling = int(kwargs.get('rolling', 20))
    cut_off = 1 - kwargs.get('stop_loss', 0.5)
    turtle_sell = functools.partial(turtle_sell_strategy, upper_sample=upper_sample,
                                    lower_sample=lower_sample, ATR_sample=ATR_sample, stop_loss=cut_off)
    # Train
    # train_strategy = Strategy(train, turtle_sell)
    # prior = prob_odd(train_strategy.kelly_table().Posterior.values[-1])
    prior = 0.5
    mv = int(kwargs.get('mv_bbh', 10))
    BBHLevel = int(kwargs.get('BBHLevel', 1))
    debug = kwargs.get('debug', False)


    test_strategy = Strategy(df=test, mv=mv, func_sell_strategy=turtle_sell)
    test_strategy.register_signal('Turtle Trading', df.Close > df.turtle_h)
    test_strategy.register_signal('Hit', df.BBH == BBHLevel)
    # test_strategy.register_signal('None', df.Close > 0)

    test_name = f'TTS_{upper_sample}_{lower_sample}_{ATR_sample}'
    # back test
    profit_table, w_daily_return = back_test(test_name, test_strategy, prior, rolling=rolling, initial_fund=100, breakdown=debug)
    s_summary = profit_table.iloc[0]

    # weight
    w_sample = 0.6
    w_profit_ratio = 0.20
    w_max_drawdown = 0.20

    sample = s_summary.Sample
    profit_sample = s_summary.ProfitSample
    loss_sample = sample - profit_sample
    sample_power = (profit_sample/loss_sample-1) * (sample/365) if (profit_sample & loss_sample) else 0.1

    profit_ratio = s_summary['ProfitRatio'] - 1

    max_drawdown = s_summary.Drawdown if not np.isnan(s_summary.Drawdown) else -1
    score = (w_sample * sample_power) + (w_profit_ratio * profit_ratio) + (w_max_drawdown * max_drawdown)

    if debug:
        print(f'Sample power: {profit_sample}/{loss_sample} - 1 * {sample} / 365 ({w_sample * sample_power:.2f}), '
              f'profit_ratio: {profit_ratio} ({(w_profit_ratio * profit_ratio):.2f}), '
              f'max_drawdown: {max_drawdown:.2f} ({w_max_drawdown*max_drawdown:.2f}), '
              f'score: {score:.2f}')
        profit_table.to_csv('performance.csv', index=False)
        print(profit_table)
        test_strategy.df.to_csv('FULLDUMP.csv', index=False)

    return score


def optimizer(train, test):
    # 定義參數空間
    params = {
        'upper_sample': (2, 120),
        'lower_sample': (2, 120),
        'ATR_sample': (2, 120),
        'stop_loss': (0, 1),
        'rolling': (30, 185),
        'BBHLevel': (1, 10),
        'mv_bbh': (3, 180),
        'vol_level': (1, 10),
    }


    # 使用貝葉斯優化進行參數優化
    optimizer = BayesianOptimization(f=target_function, pbounds=params, random_state=1)
    optimizer.maximize(init_points=3, n_iter=100)

    # 獲取最優參數組合和分數
    best_params = optimizer.max['params']
    best_score = optimizer.max['target']

    import pprint
    print(f'best_params = {pprint.pformat(best_params)}')
    print('best_score = ', best_score)
    return best_params


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    pd.set_option('display.width', 300)

    # df = pd.read_csv('data/DOT-USD.csv')
    # df = pd.read_csv('data/NEAR-USD.csv')
    df = pd.read_csv('data/BTC-USD.csv')

    # df = pd.read_csv('data/7113.KL.csv')
    # df = pd.read_csv('data/1155.KL.csv')  # Maybank, not a good deal
    # df = pd.read_csv('data/1295.KL.csv')  # pbbank, not a good deal
    # df = pd.read_csv('data/5168.KL.csv')    # Hartalega, good performance: best_params =  {'ATR_sample': 2.1402178794670643, 'lower_sample': 88.43062793247516, 'rolling': 146.2221786868714, 'upper_sample': 35.03760498083406}
    # df = pd.read_csv('data/NS8U.SI.csv')
    # df = pd.read_csv('data/5127.KL.csv')    # ARREIT
    # df = pd.read_csv('data/7106.KL.csv')    # Supermax




    df = df.dropna()

    # train, test = split_train_test(df, 0.1)
    train, test = df, df


    run_optimize = 0

    best_params = {'ATR_sample': 17.17822871698939,
                     'BBHLevel': 1.1874643019432607,
                     'lower_sample': 106.69424540413839,
                     'mv_bbh': 144.4980504213359,
                     'rolling': 154.409306948777,
                     'stop_loss': 0.47132854713381345,
                     'upper_sample': 99.4865981746286,
                     'vol_level': 6.404733566815136}

    if run_optimize:
        best_params = optimizer(train, test)
    print('--' * 100)
    best_params['debug'] = True
    target_function(**best_params)