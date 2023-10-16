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


def enrichment_daily_profit(base_df, params):
    _loss_margin = params.atr_loss_margin or 1.5
    base_df = turtle_trading(base_df, params)

    # Initialize columns if they don't exist
    for col in ["buy", "sell", "profit", "time_cost", "Matured"]:
        base_df[col] = base_df.get(col, pd.NaT if col == "Matured" else np.nan)

    resume_idx = base_df.sell.isna().idxmax()
    df = base_df.loc[resume_idx:].copy()

    # Buy daily basic when the market price is higher than Stop profit
    buy_condition = (
        df.buy.isna() & df.Stop_profit.notna() & (df.Stop_profit < df.Open.shift(-1))
    )
    df.loc[buy_condition, "buy"] = df.Open.shift(-1)

    # Sell condition:
    sell_condition = df.buy.notna() & (
        (df.Close.shift(-1) < df.Stop_profit) | (df.Close.shift(-1) < df.turtle_l)
    )

    df.loc[sell_condition, "sell"] = df.Stop_profit.where(sell_condition)
    df.loc[sell_condition, "Matured"] = pd.to_datetime(
        df.Date.shift(-1).where(sell_condition)
    )

    # Backfill sell and Matured columns
    df.sell.bfill(inplace=True)
    df.Matured.bfill(inplace=True)

    # Compute profit and time_cost columns
    profit_condition = df.buy.notna() & df.sell.notna() & df.profit.isna()
    df.loc[profit_condition, "profit"] = (df.sell / df.buy) - 1
    df.loc[profit_condition, "time_cost"] = pd.to_datetime(df.Matured) - pd.to_datetime(
        df.Date
    )

    # Clear sell and Matured values where buy is NaN
    df.loc[df.buy.isna(), ["sell", "Matured"]] = np.nan
    base_df.update(df)
    return base_df
