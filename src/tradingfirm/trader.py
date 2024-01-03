from story import IHunter
from utils import pandas_util
import pandas as pd


class GainsBag:
    def __init__(self, init_fund, coins):
        self.init_fund = self.cash = init_fund
        self.coins = coins

    def cash_in(self, cash):
        self.cash += cash


class xHunter(IHunter):
    def __init__(self, params):
        super().__init__()
        self.params = params 
        self.gains_bag = GainsBag(init_fund=400, coins=0)


    def strike_phase(self, base_df):
        print(base_df)
        return base_df