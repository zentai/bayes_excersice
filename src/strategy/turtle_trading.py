import sys
import os
from story import IStrategyScout
from utils import pandas_util
import numpy as np
import pandas as pd


def pre_process_data(df):
    COLUMNS = [
        "ATR",
        "turtle_h",
        "turtle_l",
        "Stop_profit",
        "buy",
        "sell",
        "profit",
        "time_cost",
        "Matured",
        "BuySignal",
        "Postrior",
        "Kelly",
        "p_win",
        "likelihood",
        "profit_margin",
        "loss_margin",
    ]
    missing_columns = set(COLUMNS) - set(df.columns)
    new_cols = {col: np.nan for col in missing_columns}
    if "Matured" in new_cols:
        new_cols["Matured"] = pd.NaT

    return df.assign(**new_cols)


class TurtleScout(IStrategyScout):
    def __init__(self, params, symbols, history_data=None):
        self.params = params
        self.symbols = symbols
        self.interval = "1min"
        self.base_df = history_data or pandas_util.load_symbols_from_huobi(self.symbols, self.interval)
        self.base_df = pre_process_data(self.base_df)

        self.update_idx = 0

    def update(self):
        new_data = pandas_util.get_history_stick(self.symbols, sample=1, interval=self.interval).iloc[-1]
        # new_data = self.test_df.iloc[self.update_idx].copy()
        new_data["Matured"] = pd.NaT
        self.base_df = pd.concat(
            [self.base_df, pd.DataFrame([new_data], columns=self.base_df.columns)],
            ignore_index=True
        )
        print(self.base_df)
        self.update_idx += 1

    def market_recon(self):
        self._calc_ATR()
        self._calc_profit()
        return self.base_df

    def _calc_profit(self):
        base_df = self.base_df
        _loss_margin = self.params.atr_loss_margin or 1.5

        resume_idx = base_df.sell.isna().idxmax()
        df = base_df.loc[resume_idx:].copy()

        # Buy daily basic when the market price is higher than Stop profit
        s_buy = (
            df.buy.isna()
            & df.exit_price.notna()
            & (df.exit_price < df.Open.shift(-1))
        )
        df.loc[s_buy, "buy"] = df.Open.shift(-1)
        df.loc[:, "BuySignal"] = df.High > df.turtle_h
        # Sell condition:
        s_sell = df.buy.notna() & (df.Low.shift(-1) < df.exit_price)
        df.loc[s_sell, "sell"] = df.exit_price.where(s_sell)
        df.loc[s_sell, "Matured"] = pd.to_datetime(df.Date.shift(-1).where(s_sell))

        # Backfill sell and Matured columns
        df.sell.bfill(inplace=True)
        df.Matured.bfill(inplace=True)

        # Compute profit and time_cost columns
        s_profit = df.buy.notna() & df.sell.notna() & df.profit.isna()
        df.loc[s_profit, "profit"] = (df.sell / df.buy) - 1
        df.loc[s_profit, "time_cost"] = [
            x.days
            for x in (
                pd.to_datetime(df.loc[s_profit, "Matured"])
                - pd.to_datetime(df.loc[s_profit, "Date"])
            )
        ]

        # Clear sell and Matured values where buy is NaN
        df.loc[df.buy.isna(), "sell"] = np.nan
        df.loc[df.buy.isna(), "Matured"] = pd.NaT
        base_df.update(df)
        return base_df

    def _calc_ATR(self):
        params = self.params
        base_df = self.base_df

        # performance: only re-calc nessasary part.
        idx = (
            base_df.index
            if base_df.ATR.isna().all()
            else base_df.ATR.iloc[params.ATR_sample :].isna().index
        )
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
        base_df.loc[idx, "exit_price"] = base_df[["turtle_l", "Stop_profit"]].max(
            axis=1
        )
        return base_df

    # def s_turtle_buy(base_df, params):
    #     df = _calc_ATR(base_df, params)
    #     return df.Close > df.turtle_h
