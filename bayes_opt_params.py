from bayes_opt import BayesianOptimization
from sklearn.model_selection import ParameterGrid
import functools
# from bayes_excercise import *
from bayes_kelly import *
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.filterwarnings('ignore', category=SettingWithCopyWarning)


def optimize_func(csv_path, **kwargs):
    tparam.update(kwargs)
    debug = kwargs.get('debug', False)

    df = pd.read_csv(csv_path)
    df = df.dropna()
    bkf = BayesKelly(df, name=csv_path.split('/')[-1])
    bkf.register_signal('TurtleBuy', s_turtle_buy)
    bkf.register_signal('VegasBuy', s_vegas_tunnel_buy)
    # bkf.register_signal('BBL', s_bollinger_band)
    kelly_df = bkf.bayes_update(prior=0.5, debug=debug)
    profit_table, w_daily_return, simulate_transaction_df = back_test(kelly_df, prior=0.5, breakdown=debug)

    s_summary = profit_table.iloc[0]

    # 胜率 = 获利交易数 / 总交易数
    win_rate = (simulate_transaction_df['kelly(f)'] > 0).sum() / (simulate_transaction_df['kelly(f)'] <= 0).sum()
    # win_rate = (s_summary.ProfitSample + 0.001) / (s_summary.Sample + 0.001)
    # profit_loss_ratio = s_summary.ProfitLossRatio
    # profit_loss_ratio = (simulate_transaction_df['kelly(f)'] > 0 & simulate_transaction_df['F.ProfitAmount'] > 0).mean() / (simulate_transaction_df['kelly(f)'] > 0 & simulate_transaction_df['F.ProfitAmount'] < 0).mean()
    # sample_weight = len(simulate_transaction_df) / (len(df) + 2)
    score = win_rate
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
        profit_table.to_csv(f'{bkf._name}_performance.csv', index=False)
        print(profit_table)
        kelly_df.to_csv(f'{bkf._name}_best_kelly.csv', index=False)
        print(f'created: {bkf._name}_best_kelly.csv')
        full_df = pd.merge(bkf._df, simulate_transaction_df, how='left', on=['Date'])
        full_df['TTL_signal'] = full_df.Close > full_df.turtle_h
        full_df.to_csv(f'{bkf._name}_full_df.csv', index=False)
        print(f'created: {bkf._name}_full_df.csv')
    return score


def run():
    run_optimize = 1
    best_params = {'ATR_sample': 67.45465113173290,
                     'atr_loss_margin': 1.2736213988508600,
                     'bayes_windows': 493.7536918075720,
                     'lower_sample': 17.991732615371500,
                     'upper_sample': 76.13170396012780}
    # a = '''1295.KL.csv 4952.KL.csv 5116.KL.csv 5176.KL.csv 5280.KL.csv A68U.SI.csv AW9U.SI.csv C2PU.SI.csv CMOU.SI.csv D07.SI.csv J91U.SI.csv M44U.SI.csv O87.SI.csv RW0U.SI.csv 5099.KL.csv 5120.KL.csv 5180.KL.csv 5347.KL.csv A7RU.SI.csv BMGU.SI.csv C38U.SI.csv CNNU.SI.csv D5IU.SI.csv JYEU.SI.csv ME8U.SI.csv ODBU.SI.csv S27.SI.csv TS0U.SI.csv 1F2.SI.csv 5106.KL.csv 5121.KL.csv 5212.KL.csv 6963.KL.csv ACV.SI.csv BTOU.SI.csv C61U.SI.csv COI.SI.csv D8DU.SI.csv K2LU.SI.csv MXNU.SI.csv OVQ.SI.csv S7OU.SI.csv UD1U.SI.csv 2605.TW.csv 5109.KL.csv 5123.KL.csv 5227.KL.csv 7100.KL.csv ADQU.SI.csv BUOU.SI.csv CEDU.SI.csv CRPU.SI.csv HMN.SI.csv K71U.SI.csv N2IU.SI.csv OXMU.SI.csv SK6U.SI.csv XZL.SI.csv 3182.KL.csv 5110.KL.csv 5127.KL.csv 5235SS.KL.csv 7113.KL.csv AJBU.SI.csv BWCU.SI.csv CJLU.SI.csv CWCU.SI.csv J69U.SI.csv LIW.SI.csv NS8U.SI.csv P40U.SI.csv SV3U.SI.csv 4715.KL.csv 5111.KL.csv 5130.KL.csv 5269.KL.csv A17U.SI.csv AU8U.SI.csv BYJ.SI.csv CLR.SI.csv CY6U.SI.csv J85.SI.csv M1GU.SI.csv O5RU.SI.csv Q5T.SI.csv T82U.SI.csv'''
    # a = '''1295.KL.csv 4952.KL.csv 5116.KL.csv 5176.KL.csv 5280.KL.csv A68U.SI.csv AW9U.SI.csv C2PU.SI.csv CMOU.SI.csv D07.SI.csv J91U.SI.csv M44U.SI.csv O87.SI.csv RW0U.SI.csv 5099.KL.csv 5120.KL.csv 5180.KL.csv'''
    a = '''O87.SI.csv'''
    metas = {"BTC-USD.csv": "BTC-USD",
    "ETH-USD.csv": "ETH-USD",
    "USDT-USD.csv": "USDT-USD",
    "BNB-USD.csv": "BNB-USD",
    "USDC-USD.csv": "USDC-USD",
    "XRP-USD.csv": "XRP-USD",
    "HEX-USD.csv": "HEX-USD",
    "ADA-USD.csv": "ADA-USD",
    "MATIC-USD.csv": "MATIC-USD",
    "DOGE-USD.csv": "DOGE-USD",
    "BUSD-USD.csv": "BUSD-USD",
    "SOL-USD.csv": "SOL-USD",
    "DOT-USD.csv": "DOT-USD",
    "WTRX-USD.csv": "WTRX-USD",
    "LTC-USD.csv": "LTC-USD",
    "SHIB-USD.csv": "SHIB-USD",
    "TRX-USD.csv": "TRX-USD",
    "STETH-USD.csv": "STETH-USD",
    "AVAX-USD.csv": "AVAX-USD",
    "DAI-USD.csv": "DAI-USD",
    "UNI7083-USD.csv": "UNI7083-USD",
    "LINK-USD.csv": "LINK-USD",
    "ATOM-USD.csv": "ATOM-USD",
    "WBTC-USD.csv": "WBTC-USD",
    "LEO-USD.csv": "LEO-USD",
    "OKB-USD.csv": "OKB-USD",
    "TON11419-USD.csv": "TON11419-USD",
    "ETC-USD.csv": "ETC-USD",
    "XMR-USD.csv": "XMR-USD",
    "LDO-USD.csv": "LDO-USD"}

    metas = {"BTC-USD.csv": "BTC-USD"}
    test_cases = metas.keys()
    # test_cases = ['data/S35.SI.csv', 'data/NEAR-USD.csv']

    scanning_df = pd.DataFrame([], columns=['Name', 'best_score', 'upper_sample', 'lower_sample', 'ATR_sample', 'atr_loss_margin', 'bayes_windows', 'max_invest'])
    for tname in test_cases:
        try:
            optimize_func_wrapper = functools.partial(optimize_func, csv_path=f'/Users/zen/Documents/code/FinRPG/data/history_price/{tname}')
            if run_optimize:
            # 定義參數空間
                params = {
                    'upper_sample': (2, 120),
                    'lower_sample': (2, 120),
                    'ATR_sample': (2, 120),
                    'atr_loss_margin': (1, 3),
                    'bayes_windows': (30, 720),
                    # 'max_invest': (100, 1500),
                }

                optimizer = BayesianOptimization(f=optimize_func_wrapper, pbounds=params, random_state=1)
                optimizer.maximize(init_points=5, n_iter=100)
                best_params = optimizer.max['params']
                best_score = optimizer.max['target']
                import pprint
                print(f'    best_params = {pprint.pformat(best_params)}')
                print('best_score = ', best_score)
                best_params['Name'] = tname
                best_params['best_score'] = best_score
                scanning_df = scanning_df.append(pd.Series(best_params), ignore_index=True)
            # print('--' * 100)
            best_params['debug'] = True
            optimize_func_wrapper(**best_params)
        except Exception as e:
            print(e)
    print(scanning_df)
    scanning_df.to_csv(f'scanning_{len(test_cases)}.csv', index=False)
    print(f'created: scanning_{len(test_cases)}.csv')


def portfolio():
    run_optimize = 0
    #
    best_params_df = pd.read_csv('scanning_30.csv')
    best_params_df = best_params_df[best_params_df.best_score > 0]


    metas = { name: name.replace('.csv', '') for name in best_params_df['Name'].values}
    test_cases = metas.keys()
    # test_cases = ['data/S35.SI.csv', 'data/NEAR-USD.csv']

    scanning_df = pd.DataFrame([], columns=['Name', 'best_score', 'upper_sample', 'lower_sample', 'ATR_sample', 'atr_loss_margin', 'bayes_windows', 'max_invest'])
    for tname in test_cases:
        # try:
            best_params = best_params_df[best_params_df.Name == tname].to_dict('records')[0]
            optimize_func_wrapper = functools.partial(optimize_func, csv_path=f'/Users/zen/Documents/code/FinRPG/data/history_price/{tname}')
            if run_optimize:
            # 定義參數空間
                params = {
                    # Bollinger band
                    # 'bbh_window': (5, 60),
                    # 'bbl_window': (5, 60),
                    # 'bbl_grade': (3, 10),
                    'upper_sample': (2, 120),
                    'lower_sample': (2, 120),
                    'ATR_sample': (2, 120),
                    'atr_loss_margin': (1, 3),
                    'bayes_windows': (30, 720),
                    # 'max_invest': (100, 1500),
                }

                optimizer = BayesianOptimization(f=optimize_func_wrapper, pbounds=params, random_state=1)
                optimizer.maximize(init_points=5, n_iter=100)
                best_params = optimizer.max['params']
                best_score = optimizer.max['target']
                import pprint
                print(f'    best_params = {pprint.pformat(best_params)}')
                print('best_score = ', best_score)
                best_params['Name'] = tname
                best_params['best_score'] = best_score
                scanning_df = scanning_df.append(pd.Series(best_params), ignore_index=True)
            # print('--' * 100)
            best_params['debug'] = True
            optimize_func_wrapper(**best_params)
        # except Exception as e:
        #     print(e)
    print(scanning_df)
    # scanning_df.to_csv(f'scanning_{len(test_cases)}.csv', index=False)
    # print(f'created: scanning_{len(test_cases)}.csv')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    pd.set_option('display.width', 300)
    run()
    # portfolio()
