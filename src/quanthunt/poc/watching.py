import pandas as pd

watching_list = set(['ORC-USD', 'MBOX-USD', 'POLC-USD', 'HBC-USD', 'KOK-USD', 'SAND-USD', 'INV-USD', 'POR-USD', 'AXS-USD', 'PYR-USD', 'REVV-USD', 'SD-USD', 'DFI-USD', 'WEMIX-USD', 'XCUR-USD', 'LUNC-USD', 'SOL-USD', 'CAKE-USD', 'PROM-USD', 'SDAO-USD', 'EGLD-USD', 'GRT-USD', 'FRONT-USD', 'CRU-USD', 'OPUL-USD', 'ARG-USD', 'DODO-USD', 'RNDR-USD', 'DORA-USD', 'KSM-USD', 'PBR-USD', 'XYO-USD', 'VLX-USD', 'POOLZ-USD', 'CTSI-USD', 'FUSE-USD', 'AUDIO-USD', 'YFII-USD', 'AVAX-USD', 'ADP-USD', 'CTC-USD', 'WILD-USD', 'OGN-USD', 'CVX-USD', 'CEL-USD', 'SFUND-USD', 'KCAL-USD', 'ANKR-USD', 'EDEN-USD', 'AR-USD', 'PSG-USD', 'HUNT-USD', 'MOOV-USD', 'DOGE-USD', 'BADGER-USD', 'AKT-USD', 'THETA-USD', 'METIS-USD', 'CEEK-USD', 'INJ-USD', 'KLAY-USD', 'GT-USD', 'GALA-USD', 'ABBC-USD', 'DEXE-USD', 'DKA-USD', 'MLK-USD', 'LDO-USD', 'SCRT-USD', 'MANA-USD', 'HTR-USD', 'ALGO-USD', 'UMA-USD', 'NEAR-USD', 'XRT-USD', 'ACH-USD', 'RADAR-USD', 'FTT-USD', 'TITAN-USD', 'SOLO-USD', 'STAKE-USD', 'FTM-USD', 'GHST-USD', 'XNO-USD', 'RING-USD', 'DFX-USD', 'CUBE-USD', 'BNB-USD', 'POLS-USD', 'SNX-USD', 'FX-USD', 'HBAR-USD', 'LAMB-USD', 'SWAP-USD', 'NOIA-USD', 'YFI-USD', 'SUKU-USD', 'UOS-USD', 'AAVE-USD', 'COTI-USD', 'MASK-USD', 'AQT-USD', 'AGIX-USD', 'SUSHI-USD', 'ETC-USD', 'SXP-USD', 'BAND-USD', 'ATM-USD', 'LN-USD', 'JUV-USD', 'ADA-USD', 'KAI-USD', 'JST-USD', 'WAVES-USD', 'CRO-USD', 'RLC-USD', 'NU-USD', 'MXC-USD', 'DOT-USD', 'SSV-USD', 'BNT-USD', 'KAVA-USD', 'TON-USD', 'ENJ-USD', 'LPT-USD', 'HIVE-USD', 'MPL-USD', 'ETH-USD', 'SOC-USD', 'OG-USD', 'ROUTE-USD', 'UFT-USD', 'OCEAN-USD', 'FSN-USD', 'GNO-USD', 'SC-USD', 'LRC-USD', 'TRB-USD', 'XDC-USD', 'VRA-USD', 'BTC-USD', 'HT-USD', 'WHALE-USD', 'NEO-USD', 'WBTC-USD', 'BIX-USD', 'TRX-USD', 'PRQ-USD', 'XRP-USD', 'AE-USD', 'ZRX-USD', 'ZKS-USD', 'ICX-USD', 'FET-USD', 'QTUM-USD', 'UTK-USD', 'GAL-USD', 'XLM-USD', 'XVG-USD', 'MLN-USD', 'BAT-USD', 'SOUL-USD', 'IOTX-USD', 'OMG-USD', 'MX-USD', 'PHB-USD', 'RVN-USD', 'SRM-USD', 'ORBS-USD', 'GXC-USD', 'KLV-USD', 'LINK-USD', 'NULS-USD', 'TLOS-USD', 'ZEN-USD', 'MATIC-USD', 'LTC-USD', 'ARPA-USD', 'DCR-USD', 'REN-USD', 'MTL-USD', 'ATOM-USD', 'XCAD-USD', 'WNXM-USD', 'NAS-USD', 'ERG-USD', 'ABT-USD', 'FIRO-USD', 'EOS-USD', 'NCT-USD', 'ITC-USD', 'BTM-USD', 'SNT-USD', 'GLM-USD', 'CHSB-USD'])

onedf = []
signal_in_month = []
for code in watching_list:
	df = pd.read_csv(f'reports/{code}.csv_performance.csv')
	df['Code'] = code
	onedf.append(df)

	signal_df = pd.read_csv(f'reports/{code}.csv_best_kelly.csv')
	if signal_df.iloc[-1].Date > '2023-03-16':
		print(f'======= {code} ========')
		print(signal_df.tail(1))
		signal_in_month.append(code)
combined_df = pd.concat(onedf, ignore_index=True)
col_order = ['Code'] + [col for col in combined_df.columns if col != 'Code']
combined_df = combined_df[col_order]
print(combined_df)
combined_df.to_csv('reports/best_performance.csv')
print(f'generated reports/best_performance.csv')

print(f'Buy Signals:')
print(combined_df[combined_df.Code.isin(signal_in_month)])
combined_df[combined_df.Code.isin(signal_in_month)].to_csv('reports/best_Signal_in_the_month.csv')
print(f'reports/best_Signal_in_the_month.csv')
for code in signal_in_month:
	print(f'{code}.csv_full_df.csv')
	print(f'{code}.csv_performance.csv')
	print(f'{code}.csv_best_kelly.csv')
