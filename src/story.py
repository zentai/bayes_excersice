import sys
import os
import time 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

DEBUG_COL = [
                    "Date",
                    # "Open",
                    # "High",
                    # "Low",
                    "Close",
                    # "BuySignal",
                    # "Stop_profit",
                    "exit_price",
                    "Matured",
                    "time_cost",
                    "buy",
                    "sell",
                    "profit",
                    "Kelly",
                    "Postrior",
                    "D P/L",
                    "likelihood",
                    "profit_margin",
                    "loss_margin",
                ] 

class IStrategyScout(ABC):
    @abstractmethod
    def market_recon(self, mission_blueprint):
        pass

    @abstractmethod
    def update(self):
        pass


class IEngine(ABC):
    @abstractmethod
    def generate_hunt_plan(self, recon_report):
        pass


@dataclass
class HuntingStory:
    scout: IStrategyScout
    engine: IEngine
    # hunter: "xHunter"
    # gains_bag: "GainsBag"
    # mission_blueprint: "MissionBlueprint"
    # capital_trap: "CapitalTrap"

    def start(self):
        recon_report = self.scout.market_recon()
        hunt_plan = self.engine.generate_hunt_plan(recon_report)
        print(
            hunt_plan[DEBUG_COL][-50:]
        )
        time.sleep(60)
        self.scout.update()
        return hunt_plan
        # hunt_plan.to_csv("hunt.csv")
        # print("hunt.csv")
        # self.hunter.execute_trade(hunt_plan, self.capital_trap)
        # trade_result = self.capital_trap.monitor_trade()
        # self.gains_bag.update(trade_result)
        # self.hunter.observe_trade(self.capital_trap)
        # self.hunter.review_mission()


@dataclass
class StrategyParam:
    ATR_sample: int = 7
    atr_loss_margin: float = 1.5
    bayes_windows: int = 30
    lower_sample: int = 7
    upper_sample: int = 7

    def __post_init__(self):
        self.ATR_sample = int(self.ATR_sample)
        self.bayes_windows = int(self.bayes_windows)
        self.lower_sample = int(self.lower_sample)
        self.upper_sample = int(self.upper_sample)


if __name__ == "__main__":
    import sys
    import os

    from settings import DATA_DIR, SRC_DIR, REPORTS_DIR
    from strategy.turtle_trading import TurtleScout
    from engine.probabilistic_engine import BayesianEngine

    params = {
        "ATR_sample": 30,
        "atr_loss_margin": 2,
        "bayes_windows": 30,
        "lower_sample": 30,
        "upper_sample": 30,
    }
    sp = StrategyParam(**params)
    scout = TurtleScout(params=sp, symbols="btcusdt")
    engine = BayesianEngine(params=sp)
    # hunter = xHunter()
    # gains_bag = GainsBags(init_fund=100, position=0)

    story = HuntingStory(scout, engine)
    for i in range(2000):
        hunt_plan = story.start()
        hunt_plan[
                    [
                        "Date",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "BuySignal",
                        "Stop_profit",
                        "exit_price",
                        "buy",
                        "sell",
                        "profit",
                        "turtle_l",
                        "turtle_h",
                        "Matured",
                        "time_cost",
                        "Kelly",
                        "Postrior",
                        "Sig P/L",
                        "D P/L",
                        "likelihood",
                        "profit_margin",
                        "loss_margin",
                    ]
                ].to_csv("hunt.csv")
    print("hunt.csv")

