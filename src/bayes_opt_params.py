import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from settings import DATA_DIR, SRC_DIR, REPORTS_DIR

from bayes_opt import BayesianOptimization
from sklearn.model_selection import ParameterGrid
import functools
# from bayes_excercise import *
from bayes_kelly import enrichment_daily_profit
from bayes_kelly import s_turtle_buy
from bayes_kelly import pick_dates
from bayes_kelly import enrichment_temp_close
from bayes_kelly import calc_likelihood
from bayes_kelly import kelly_formular
from bayes_kelly import adjust_like
from bayes_kelly import back_test

from bayes_kelly import BayesKelly
from bayes_kelly import StrategyParam
import pandas as pd
import numpy as np
import warnings


def optimize_func(code, **kwargs):
    sp = StrategyParam(**kwargs)
    debug = False
    # debug = kwargs.get('debug', False)

    def load_and_split_data(code, train_ratio=0.4):
        df = pd.read_csv(f'{DATA_DIR}/{code}.csv')
        df = df.dropna()
        size = len(df)
        train_size = int(train_ratio * size)
        train_df = df[:train_size]
        test_df = df[train_size:]
        return train_df, test_df

    train_df, _ = load_and_split_data(code)

    bkf = BayesKelly(train_df, sp)
    bkf.register_signal('TurtleBuy', s_turtle_buy)
    kelly_df = bkf.bayes_update(prior=0.1, debug=True)

    profit_table, w_daily_return, simulate_transaction_df = back_test(kelly_df, breakdown=False)
    s_summary = profit_table.iloc[0]

    # 胜率 = 获利交易数 / 总交易数
    # win_rate = (simulate_transaction_df['kelly(f)'] > 0).sum() / (simulate_transaction_df['kelly(f)'] <= 0).sum()
    # win_rate = (simulate_transaction_df['kelly(f)'] > 0).sum() / len(simulate_transaction_df['kelly(f)'])

    # win_rate = (s_summary.ProfitSample + 0.001) / (s_summary.Sample + 0.001)
    profit_loss_ratio = s_summary.ProfitLossRatio
    score = profit_loss_ratio
    # profit_loss_ratio = (simulate_transaction_df['kelly(f)'] > 0 & simulate_transaction_df['F.ProfitAmount'] > 0).mean() / (simulate_transaction_df['kelly(f)'] > 0 & simulate_transaction_df['F.ProfitAmount'] < 0).mean()
    # sample_weight = len(simulate_transaction_df) / (len(df) + 2)
    # score = s_summary.SortinoRatio if ~np.isnan(s_summary.SortinoRatio) else 0
    # score = score if ~np.isinf(score) else 0
    # print(f'{win_rate} * {sample_weight} * {profit_loss_ratio} * {sample_weight} ')
    '''
    # weight
    w_sample = 0.6
    w_profit_ratio = 0.20
    w_max_drawdown = 0.20

    sample = s_summary.Sample + 0.001
    profit_sample = s_summary.ProfitSample + 0.001
    loss_sample = sample - profit_sample
    sample_power = (profit_sample/loss_sample-1) * (sample/365)
    # sample_power = (profit_sample/loss_sample-1) * (sample/365) if (profit_sample and loss_sample) else 0.1

    profit_ratio = s_summary['ProfitRatio'] - 1

    max_drawdown = s_summary.Drawdown if not np.isnan(s_summary.Drawdown) else -1
    score = (w_sample * sample_power) + (w_profit_ratio * profit_ratio) + (w_max_drawdown * max_drawdown)
    '''

    if debug:
        # print(f'Sample power: {profit_sample}/{loss_sample} - 1 * {sample} / 365 ({w_sample * sample_power:.2f}), '
        #       f'profit_ratio: {profit_ratio} ({(w_profit_ratio * profit_ratio):.2f}), '
        #       f'max_drawdown: {max_drawdown:.2f} ({w_max_drawdown*max_drawdown:.2f}), '
        #       f'score: {score:.2f}')
        code = csv_path.split('/')[-1].replace('.csv', '')
        profit_table.to_csv(f'{REPORTS_DIR}/{code}_performance.csv', index=False)
        print(profit_table)
        full_df = pd.merge(base_df, simulate_transaction_df, how='left', on=['Date'])
        full_df['TTL_signal'] = full_df.Close > full_df.turtle_h
        full_df.to_csv(f'{REPORTS_DIR}/{code}_full_df.csv', index=False)
        print(f'created: {REPORTS_DIR}/{code}_full_df.csv')
    return score


def run():
    run_optimize = 1
    best_params = { 'ATR_sample': 67.45465113173290,
                    'atr_loss_margin': 1.2736213988508600,
                    'bayes_windows': 493.7536918075720,
                    'lower_sample': 17.991732615371500,
                    'upper_sample': 76.13170396012780 
                }

    metas = {
        # 'P9D.SI.csv': 'P9D.SI',
        # '6888.KL.csv': '6888.KL',
        '5148.KL.csv': '5148.KL',    # UEMS
        # 'NVDA.csv': 'NVDA',
        # 'BTC-USD.csv': 'BTC-USD',
        # 'E28.SI.csv': 'E28.SI',
    }

    test_cases = metas.keys()

    scanning_df = pd.DataFrame([], columns=['Name', 'best_score', 'upper_sample', 'lower_sample', 'ATR_sample', 
                                            'atr_loss_margin', 'bayes_windows', 'max_invest'])
    
    for files, tname in metas.items():
        print(tname)
        try:
            optimize_func_wrapper = functools.partial(optimize_func, code=tname)

            params = {
                'ATR_sample': (2, 900),
                'atr_loss_margin': (1, 5),
                'bayes_windows': (3, 345),
                'lower_sample': (2, 900),
                'upper_sample': (2, 900),
            }

            params = {
                'ATR_sample': (84, 884),
                'atr_loss_margin': (1, 5),
                'bayes_windows': (114, 428),
                'lower_sample': (221, 891),
                'upper_sample': (120, 485),
            }

            params = {
                'ATR_sample': (270, 754),
                'atr_loss_margin': (1.773, 4.891),
                'bayes_windows': (117, 345),
                'lower_sample': (241, 890),
                'upper_sample': (153, 413),
            }

            params = {
                'ATR_sample': (386, 725),
                'atr_loss_margin': (3.352, 4.322),
                'bayes_windows': (163, 242),
                'lower_sample': (762, 889),
                'upper_sample': (195, 367),
            }

            params = {
                'ATR_sample': (488, 552),
                'atr_loss_margin': (3.3, 4.0),
                'bayes_windows': (164, 199),
                'lower_sample': (872, 888),
                'upper_sample': (195, 196),
            }

            params = {
                'ATR_sample': (488, 552),
                'atr_loss_margin': (3.361, 4.322),
                'bayes_windows': (164, 199),
                'lower_sample': (872, 888),
                'upper_sample': (195, 196),
            }
            optimizer = BayesianOptimization(f=optimize_func_wrapper, pbounds=params, random_state=1, allow_duplicate_points=True)
            optimizer.maximize(init_points=5, n_iter=100)
            best_params = optimizer.max['params']
            best_score = optimizer.max['target']
            import pprint
            print(f'    best_params = {pprint.pformat(best_params)}')
            print('best_score = ', best_score)
            best_params['Name'] = tname
            best_params['best_score'] = best_score
            scanning_df = scanning_df.append(pd.Series(best_params), ignore_index=True)
            best_params['debug'] = False
            optimize_func_wrapper(**best_params)
        except Exception as e:
            print(e)
    print(scanning_df)
    scanning_df.to_csv(f'{REPORTS_DIR}/scanning_{len(test_cases)}.csv', index=False)
    print(f'created: {REPORTS_DIR}/scanning_{len(test_cases)}.csv')



if __name__ == '__main__':
    run()
