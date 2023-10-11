def turtle_trading(base_df):
    upper_sample = int(tparam.get('upper_sample', 20) )
    lower_sample = int(tparam.get('lower_sample', 10) )
    ATR_sample = int(tparam.get('ATR_sample', 20) )
    loss_margin = tparam.get('atr_loss_margin', 1.5)

    is_scratch = 'ATR' not in base_df.columns
    windows = len(base_df) if is_scratch else np.max([upper_sample, lower_sample, ATR_sample]) + 1

    # performance: only re-calc nessasary part.
    df = base_df.iloc[-windows:].copy()
    idx = df.index if is_scratch else df[np.isnan(df['ATR'])].index
    df = df.assign(turtle_h = df.Close.shift(1).rolling(upper_sample).max())
    df = df.assign(turtle_l = df.Close.shift(1).rolling(lower_sample).min())
    df = df.assign(h_l = df.High - df.Low)
    df = df.assign(c_h = (df.Close.shift(1)-df.High).abs())
    df = df.assign(c_l = (df.Close.shift(1)-df.Low).abs())
    df = df.assign(TR = df[['h_l', 'c_h', 'c_l']].max(axis=1))
    df = df.assign(ATR = (df.TR.rolling(ATR_sample).sum()/ATR_sample))
    

    # copy value to base_df
    base_df.loc[idx, 'turtle_h'] = df.loc[idx, 'turtle_h']
    base_df.loc[idx, 'turtle_l'] = df.loc[idx, 'turtle_l']
    base_df.loc[idx, 'ATR'] = df.loc[idx, 'ATR']
    base_df.loc[idx, 'Stop_profit'] = df.Close.shift(1) - df.ATR.shift(1) * loss_margin
    return base_df


def s_turtle_buy(base_df):
    df = turtle_trading(base_df)
    return df.Close > df.turtle_h


def enrichment_daily_profit(base_df):
    _loss_margin = tparam.get('atr_loss_margin', 1.5)
    base_df = turtle_trading(base_df)

    sell = []
    time_cost = []

    is_scratch = 'buy' not in base_df.columns
    if is_scratch:
        base_df = base_df.assign(buy=np.nan)
        base_df = base_df.assign(sell=np.nan)
        base_df = base_df.assign(time_cost=np.nan)

    sell_idx = base_df[np.isnan(base_df['sell'])].index
    windows = -len(base_df) if is_scratch else sell_idx[0]

    df = base_df.loc[windows:].copy()
    buy_idx = df.index if is_scratch else df[np.isnan(df['buy'])].index
    
    # Stop_profit > Open(tomorrow), we buy
    df.loc[buy_idx, 'buy'] = df.Open.shift(-1).where(df['Stop_profit'].notna() & (df['Stop_profit'] < df.Open.shift(-1)), np.nan)
    
    # Close(tomorrow) < Stop_profit(today) | Close(tomorrow) < turtle low (tomorrow). we sell 
    sell_point = df.Stop_profit.where((df.Close.shift(-1) < df.Stop_profit) | (df.Close.shift(-1) < df.turtle_l.shift(-1)), np.nan)
    df.loc[sell_point.index, 'sell'] = sell_point

    ic(sell_point.index)
    
    df.sell.fillna(method='bfill', inplace=True)
    df.loc[df['buy'].isna(), 'sell'] = np.nan

    
    ic(df[sell_point.notna()])
    ic(df.loc[100:159])

    
    stop_profits = []
    for i, _v in enumerate(df.loc[sell_idx][['buy', 'sell', 'time_cost', 'ATR', 'Stop_profit']].values):
        _buy, _sell, _time_cost, _buy_atr, stop_profit = _v
        _pre_atr = _buy_atr
        _pre_close = _buy
        if np.isnan(_buy):
            sell.append(_sell)
            time_cost.append(_time_cost)
            continue
        if not np.isnan(_sell):
            sell.append(_sell)
            time_cost.append(_time_cost)
            continue

        sell_point = None
        days = 0
        for j, v in enumerate(df[['Close', 'turtle_l', 'ATR']].iloc[i+1:].values):
            _close, _turtle_low, _atr = v
            sell_point, days = (_close, j) if (_close < stop_profit) or (_close < _turtle_low) else (None, None)
            if sell_point:
                break
            _pre_close = _close
            _pre_atr = _atr
        if sell_point:
            sell.append(sell_point)
            stop_profits.append(stop_profit)
            time_cost.append(days+1)
        else:
            sell.append(np.nan)
            stop_profits.append(stop_profit)
            time_cost.append(np.nan)

    # print(f'{i} - {len(buy_idx)} - {len(sell_idx)} - {len(sell)}')
    df.loc[sell_idx, 'sell'] = sell
    df.loc[sell_idx, 'time_cost'] = time_cost
    df.loc[sell_idx, 'Stop_profit'] = stop_profits
    df.loc[sell_idx, 'Matured'] = pd.to_datetime(df.loc[sell_idx, 'Date']) + pd.to_timedelta(df.loc[sell_idx, 'time_cost'], 'm')
    df.loc[sell_idx, 'profit'] = (df.loc[sell_idx, 'sell'] / df.loc[sell_idx, 'buy']) - 1

    base_df.loc[buy_idx, 'buy'] = df.loc[buy_idx, 'buy']
    base_df.loc[sell_idx, 'sell'] = df.loc[sell_idx, 'sell']
    base_df.loc[sell_idx, 'time_cost'] = df.loc[sell_idx, 'time_cost']
    base_df.loc[sell_idx, 'Matured'] = df.loc[sell_idx, 'Matured']
    base_df.loc[sell_idx, 'Stop_profit'] = df.loc[sell_idx, 'Stop_profit']
    base_df.loc[sell_idx, 'profit'] = df.loc[sell_idx, 'profit']
    return base_df
