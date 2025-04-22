import numpy as np
import pandas as pd
from dataclasses import dataclass, field

TURTLE_COLUMNS = [
    "ATR",
    "turtle_h",
    "turtle_l",
    "exit_price",
    "buy",
    "sell",
    "profit",
    "BuySignal",
    "P/L",
]


def equip_fields(df):
    new_cols = {col: np.nan for col in TURTLE_COLUMNS}
    return df.assign(**new_cols)


@dataclass
class StrategyParam:
    ATR_sample: int = 7
    atr_loss_margin: float = 1.5
    lower_sample: int = 7
    upper_sample: int = 7

    def __post_init__(self):
        self.ATR_sample = int(self.ATR_sample)
        self.lower_sample = int(self.lower_sample)
        self.upper_sample = int(self.upper_sample)


class TurtleScout:
    def __init__(self, params):
        self.params = params

    def _calc_profit(self, base_df):

        # calculate exit price
        surfing_df = base_df.copy()
        idx = surfing_df.exit_price.isna().index
        surfing_df.loc[idx, "exit_price"] = (
            surfing_df.Close.shift(1)
            - surfing_df.ATR.shift(1) * self.params.atr_loss_margin
        )
        base_df.update(surfing_df)
        resume_idx = base_df.sell.isna().idxmax()
        df = base_df.loc[resume_idx:].copy()
        df = df[df.exit_price.notna()]

        # BuySignal strategy
        s_buy = df.buy.isna()
        df.loc[s_buy, "buy"] = df.Close
        df.loc[:, "BuySignal"] = df.High > df.turtle_h

        # Sell strategy:
        s_sell = df.buy.notna() & (df.Low < df.exit_price)
        df.loc[s_sell, "sell"] = df.exit_price.where(s_sell)
        df.sell.bfill(inplace=True)

        # profit and loss
        s_profit = df.buy.notna() & df.sell.notna() & df.profit.isna()
        df.loc[s_profit, "profit"] = (df.sell / df.buy) - 1
        df.loc[s_profit, "P/L"] = (df.sell - df.buy) / (
            df.ATR * self.params.atr_loss_margin
        )

        df.loc[df.buy.isna(), "sell"] = np.nan

        base_df.update(df)
        return base_df

    def _calc_ATR(self, base_df):
        ATR_sample = self.params.ATR_sample
        upper_sample = self.params.upper_sample
        lower_sample = self.params.lower_sample

        df = base_df.copy()
        idx = df.ATR.isna().index
        df.loc[idx, "turtle_h"] = df.High.shift(1).rolling(upper_sample).max()
        df.loc[idx, "turtle_l"] = df.Low.shift(1).rolling(lower_sample).min()
        df.loc[idx, "h_l"] = df.High - df.Low
        df.loc[idx, "c_h"] = (df.Close.shift(1) - df.High).abs()
        df.loc[idx, "c_l"] = (df.Close.shift(1) - df.Low).abs()
        df.loc[idx, "TR"] = df[["h_l", "c_h", "c_l"]].max(axis=1)
        df.loc[idx, "ATR"] = df["TR"].rolling(ATR_sample).mean()
        df.loc[idx, "ATR_STDV"] = df["TR"].rolling(ATR_sample).std()
        base_df.update(df)
        return base_df

    def market_recon(self, base_df):
        base_df = equip_fields(base_df)
        base_df = self._calc_ATR(base_df)
        base_df = self._calc_profit(base_df)
        return base_df


if __name__ == "__main__":
    df = pd.read_csv(f"btcusdt.csv")
    params = {
        # Buy
        "ATR_sample": 15,
        "lower_sample": 15,
        "upper_sample": 15,
        # Sell
        "atr_loss_margin": 2,
    }
    sp = StrategyParam(**params)
    scout = TurtleScout(params=sp)
    df = scout.market_recon(df)
    df.to_csv("report.csv", index=False)
    print(df[df.sell.notna()])
