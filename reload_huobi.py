import sys
import datetime
import logging
import json
import time

from pandas import json_normalize
from yahoofinancials import YahooFinancials

def fetch(profolio=[]):
    logger = logging.getLogger(__name__)

    f = open('huobi_symbol.json')
    data = json.load(f)
    from pprint import pprint as pp

    symbols = []
    for k in data['data']:
        symbol = k['symbol']
        if 'usdt' in symbol:
            print(f"{symbol} - {symbol.replace('usdt', '-usd').upper()}")
            symbols.append(symbol.replace('usdt', '-usd').upper())

    print(f'Start process {len(symbols)} datas.')
    for i, symbol in enumerate(symbols):
        # code = meta['code']
        # start = str(meta['start'])
        code = symbol
        fname = f'{symbol}.csv'
        start = str('2014-01-01')
        end = datetime.datetime.today().strftime('%Y-%m-%d')

        yahoo_finance = YahooFinancials(code)
        daily_price = yahoo_finance.get_historical_price_data(start, end, 'daily')
        df = None
        try:
            df = json_normalize(daily_price[code]['prices'])
        except Exception as e:
            print(code, daily_price[code])
            print(daily_price)
            logger.error(e)
            continue
        df = df.drop(['date'], axis=1)
        df = df.rename(index=str, columns={'formatted_date': 'Date',
                                           'open': 'Open',
                                           'high': 'High',
                                           'low': 'Low',
                                           'close': 'Close',
                                           'adjclose': 'Adj Close',
                                           'volume': 'Volume',
                                           })
        df = df.fillna(method='ffill')
        df = df.round(decimals=2)

        df.to_csv(path_or_buf='data/%s' % fname,
                  index=False,
                  columns=['Date', 'Open', 'High', 'Low', 'Close',
                           'Adj Close', 'Volume'])

        # daily_dividend = yahoo_finance.get_daily_dividend_data(start, end)
        # df = None
        # try:
        #     df = json_normalize(daily_dividend[code])
        # except Exception as e:
        #     print(code, daily_dividend[code])
        #     logger.error(e)
        #     continue
        # df = df.drop(['date'], axis=1)
        # df = df.rename(index=str, columns={'formatted_date': 'Date',
        #                                    'amount': 'Dividends',
        #                                    })
        # df = df.fillna(method='ffill')
        # df = df.round(decimals=6)
        # df.to_csv(path_or_buf='data/dividends/%s' % fname,
        #           index=False,
        #           columns=['Date', 'Dividends'])
        print(f'[{i}] Download: {fname}')
        logger.info('Download: {fname}'.format(fname=fname))
        time.sleep(1)

if __name__ == '__main__':

    fetch()