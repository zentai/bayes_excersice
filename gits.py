# Function to calculate the likelihood of profit > 0.5 within a window
def calculate_likelihood(window):
    sorted_profits = np.sort(window)
    total_count = len(sorted_profits)
    count_greater_than_05 = np.sum(sorted_profits > 0.1)
    likelihood = count_greater_than_05 / float(total_count-count_greater_than_05+epsilon)
    print(f'{likelihood} = {count_greater_than_05} / {float(total_count-count_greater_than_05+epsilon)}')
    return likelihood


def calc_confidence_betratio(base_df, signal_func, prior=0.5, debug=False):
    params = StrategyParams(ATR_sample=20, atr_loss_margin=2, bayes_windows=20, lower_sample=20, upper_sample=20, max_invest=100)
    base_df = enrichment_daily_profit(base_df, params)
    df = enrichment_temp_close(base_df)
    df = df[s_turtle_buy]

    # Use rolling window to calculate the likelihood and create a new column
    df['likelihood'] = df.profit.rolling(window=params.bayes_windows).apply(calculate_likelihood, raw=True)

    # Initialize a new column for storing posterior odds
    df['posterior_odds'] = np.nan

    # Initialize the first odds as 1
    initial_odds = 1.0
    forgetting_factor = 0.1

    # Update the posterior odds based on the likelihood where it's not NaN
    for idx, row in df.loc[df['likelihood'].notna()].iterrows():
        # Update the odds
        new_odds = forgetting_factor * initial_odds + (1 - forgetting_factor) * row['likelihood']
        
        # Store the new posterior odds in the DataFrame
        df.loc[idx, 'posterior_odds'] = new_odds
        
        # Update the initial odds for the next iteration
        initial_odds = new_odds

    # Convert updated odds back to probability and store in a new column
    df['posterior'] = df['posterior_odds'] / (1 + df['posterior_odds'])

    print(df[-60:])
    df.to_csv('C:\\Users\\Zen\\Documents\\GitHub\\bayes_excersice\\reports\\dump.csv')
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['posterior'], marker='o', linestyle='-')
    plt.plot(df.index, df['profit'], marker='^', linestyle=':')
    plt.xlabel('Index')
    plt.ylabel('Likelihood of Profit > 0')
    plt.title('Likelihood of Profit > 0 Over Time')
    plt.grid(True)
    plt.show()

def turtle_trading(base_df, params):
    # Initialize columns if they don't exist
    for col in ["ATR", "turtle_h", "turtle_l", "Stop_profit"]:
        base_df[col] = base_df.get(col, np.nan)

    # performance: only re-calc nessasary part.
    # start_idx = base_df.ATR.isna().idxmax()
    idx = (
        base_df.index
        if base_df.ATR.isna().all()
        else base_df.ATR.iloc[params.ATR_sample :].isna().index
    )

    ic("---> before turtle trading ", base_df[:59])
    base_df.loc[idx, "turtle_h"] = (
        base_df.Close.shift(1).rolling(params.upper_sample).max()
    )
    base_df.loc[idx, "turtle_l"] = (
        base_df.Close.shift(1).rolling(params.lower_sample).min()
    )
    base_df.loc[idx, "h_l"] = base_df.High - base_df.Low
    base_df.loc[idx, "c_h"] = (base_df.Close.shift(1) - base_df.High).abs()
    base_df.loc[idx, "c_l"] = (base_df.Close.shift(1) - base_df.Low).abs()
    base_df.loc[idx, "TR"] = base_df[["h_l", "c_h", "c_l"]].max(axis=1)
    base_df.loc[idx, "ATR"] = base_df["TR"].rolling(params.ATR_sample).mean()
    base_df.loc[idx, "Stop_profit"] = (
        base_df.Close.shift(1) - base_df.ATR.shift(1) * params.atr_loss_margin
    )
    ic("---> after turtle trading ", base_df[:59])
    # copy value to base_df
    # columns_to_update = ["turtle_h", "turtle_l", "ATR", "Stop_profit"]
    # base_base_df.loc[idx, columns_to_update] = base_df.loc[idx, columns_to_update]

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
