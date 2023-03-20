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
        if symbol not in set(['ORC-USD', 'MBOX-USD', 'POLC-USD', 'HBC-USD', 'KOK-USD', 'SAND-USD', 'INV-USD', 'POR-USD', 'AXS-USD', 'PYR-USD', 'REVV-USD', 'SD-USD', 'DFI-USD', 'WEMIX-USD', 'XCUR-USD', 'LUNC-USD', 'SOL-USD', 'CAKE-USD', 'PROM-USD', 'SDAO-USD', 'EGLD-USD', 'GRT-USD', 'FRONT-USD', 'CRU-USD', 'OPUL-USD', 'ARG-USD', 'DODO-USD', 'RNDR-USD', 'DORA-USD', 'KSM-USD', 'PBR-USD', 'XYO-USD', 'VLX-USD', 'POOLZ-USD', 'CTSI-USD', 'FUSE-USD', 'AUDIO-USD', 'YFII-USD', 'AVAX-USD', 'ADP-USD', 'CTC-USD', 'WILD-USD', 'OGN-USD', 'CVX-USD', 'CEL-USD', 'SFUND-USD', 'KCAL-USD', 'ANKR-USD', 'EDEN-USD', 'AR-USD', 'PSG-USD', 'HUNT-USD', 'MOOV-USD', 'DOGE-USD', 'BADGER-USD', 'AKT-USD', 'THETA-USD', 'METIS-USD', 'CEEK-USD', 'INJ-USD', 'KLAY-USD', 'GT-USD', 'GALA-USD', 'ABBC-USD', 'DEXE-USD', 'DKA-USD', 'MLK-USD', 'LDO-USD', 'SCRT-USD', 'MANA-USD', 'HTR-USD', 'ALGO-USD', 'UMA-USD', 'NEAR-USD', 'XRT-USD', 'ACH-USD', 'RADAR-USD', 'FTT-USD', 'TITAN-USD', 'SOLO-USD', 'STAKE-USD', 'FTM-USD', 'GHST-USD', 'XNO-USD', 'RING-USD', 'DFX-USD', 'CUBE-USD', 'BNB-USD', 'POLS-USD', 'SNX-USD', 'FX-USD', 'HBAR-USD', 'LAMB-USD', 'SWAP-USD', 'NOIA-USD', 'YFI-USD', 'SUKU-USD', 'UOS-USD', 'AAVE-USD', 'COTI-USD', 'MASK-USD', 'AQT-USD', 'AGIX-USD', 'SUSHI-USD', 'ETC-USD', 'SXP-USD', 'BAND-USD', 'ATM-USD', 'LN-USD', 'JUV-USD', 'ADA-USD', 'KAI-USD', 'JST-USD', 'WAVES-USD', 'CRO-USD', 'RLC-USD', 'NU-USD', 'MXC-USD', 'DOT-USD', 'SSV-USD', 'BNT-USD', 'KAVA-USD', 'TON-USD', 'ENJ-USD', 'LPT-USD', 'HIVE-USD', 'MPL-USD', 'ETH-USD', 'SOC-USD', 'OG-USD', 'ROUTE-USD', 'UFT-USD', 'OCEAN-USD', 'FSN-USD', 'GNO-USD', 'SC-USD', 'LRC-USD', 'TRB-USD', 'XDC-USD', 'VRA-USD', 'BTC-USD', 'HT-USD', 'WHALE-USD', 'NEO-USD', 'WBTC-USD', 'BIX-USD', 'TRX-USD', 'PRQ-USD', 'XRP-USD', 'AE-USD', 'ZRX-USD', 'ZKS-USD', 'ICX-USD', 'FET-USD', 'QTUM-USD', 'UTK-USD', 'GAL-USD', 'XLM-USD', 'XVG-USD', 'MLN-USD', 'BAT-USD', 'SOUL-USD', 'IOTX-USD', 'OMG-USD', 'MX-USD', 'PHB-USD', 'RVN-USD', 'SRM-USD', 'ORBS-USD', 'GXC-USD', 'KLV-USD', 'LINK-USD', 'NULS-USD', 'TLOS-USD', 'ZEN-USD', 'MATIC-USD', 'LTC-USD', 'ARPA-USD', 'DCR-USD', 'REN-USD', 'MTL-USD', 'ATOM-USD', 'XCAD-USD', 'WNXM-USD', 'NAS-USD', 'ERG-USD', 'ABT-USD', 'FIRO-USD', 'EOS-USD', 'NCT-USD', 'ITC-USD', 'BTM-USD', 'SNT-USD', 'GLM-USD', 'CHSB-USD', 'NEXO-USD', 'WAXP-USD', 'STEEM-USD', 'STORJ-USD', 'ONT-USD', 'ELA-USD', 'XTZ-USD', 'XEM-USD', 'DUSK-USD', 'MKR-USD', 'BSV-USD', 'SPA-USD', 'REVO-USD', 'CAN-USD', 'CHZ-USD', 'WICC-USD', 'ANV-USD', 'XMR-USD', 'ZIL-USD', 'RSR-USD', 'VET-USD', 'BETH-USD', 'TRAC-USD', 'ALICE-USD', 'DOCK-USD', 'BCH-USD', 'PHA-USD', 'ANC-USD', 'AAC-USD', 'LINA-USD', 'SYS-USD', 'ANT-USD', 'VISION-USD', 'YAM-USD', '1INCH-USD', 'FIL-USD', 'DIA-USD', 'REQ-USD', 'IDEX-USD', 'CELO-USD', 'MCO-USD']):
            continue
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