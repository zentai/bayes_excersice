from bayes_opt import BayesianOptimization
from sklearn.model_selection import ParameterGrid
import functools
from bayes_excercise import *
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
def optimizer(train, test):
    # 定義目標函數
    def target_function(upper_sample, lower_sample, ATR_sample):

        # 在這裡使用參數進行回測，計算績效分數
        upper_sample = int(upper_sample)
        lower_sample = int(lower_sample)
        ATR_sample = int(ATR_sample)
        turtle_sell = functools.partial(turtle_sell_strategy, upper_sample=upper_sample,
                                        lower_sample=lower_sample, ATR_sample=ATR_sample)
        s = Strategy(train, turtle_sell)
        kelly_df = s.kelly_table(fund=100)

        test_strategy = Strategy(test, turtle_sell)
        test_name = f'TTS_{upper_sample}_{lower_sample}_{ATR_sample}'
        # back test
        profit_table, w_daily_return = back_test(test_name, test_strategy, kelly_df, breakdown=['tt_buy'])
        return w_daily_return

    # 定義參數空間
    params = {
        'upper_sample': (2, 365),
        'lower_sample': (2, 365),
        'ATR_sample': (2, 365),
    }

    # 使用貝葉斯優化進行參數優化
    optimizer = BayesianOptimization(f=target_function, pbounds=params, random_state=1)
    optimizer.maximize(init_points=3, n_iter=500)

    # 獲取最優參數組合和分數
    best_params = optimizer.max['params']
    best_score = optimizer.max['target']
    print('Best params:', best_params)
    print('Best score:', best_score)

if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    pd.set_option('display.width', 300)
    # df = pd.read_csv('data/DOT-USD.csv')
    df = pd.read_csv('data/NEAR-USD.csv')
    # df = pd.read_csv('data/7113.KL.csv')
    df = df.dropna()
    train, test = split_train_test(df, 0.2)
    print(test.head())
    optimizer(train, test)