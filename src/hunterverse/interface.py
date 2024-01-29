import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataclasses import dataclass
from abc import ABC, abstractmethod

INTERVAL_TO_MIN = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 1 * 60,
    "4hour": 4 * 60,
    "1day": 24 * 60,
    "local": 0.001,
}

class IMarketSensor(ABC):
    def __init__(self, symbol, interval):
        self.symbol = symbol
        self.interval = interval
        self.interval_min = INTERVAL_TO_MIN.get(interval)

    @abstractmethod
    def scan(self, limits):
        pass

    @abstractmethod
    def fetch(self, base_df):
        pass

    def left(self):
        return 0


class IStrategyScout(ABC):
    @abstractmethod
    def market_recon(self, mission_blueprint):
        pass


class IEngine(ABC):
    @abstractmethod
    def hunt_plan(self, base_df):
        pass


class IHunter(ABC):
    @abstractmethod
    def strike_phase(self, base_df):
        pass


@dataclass
class Symbol:
    name: str
    amount_prec: float = 4
    price_prec: float = 4

    def __post_init__(self):
        from huobi.client.generic import GenericClient

        client = GenericClient()
        for item in client.get_exchange_symbols():
            if item.symbol == self.name:
                self.amount_prec = item.amount_precision
                self.price_prec = item.price_precision

    def _round_down(self, number, prec):
        return round(
            int(number * 10**prec) / 10**prec,
            prec,
        )

    def round_price(self, price):
        return self._round_down(price, self.price_prec)

    def round_amount(self, amount):
        return self._round_down(amount, self.amount_prec)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.name


@dataclass
class StrategyParam:
    symbol: str
    ATR_sample: int = 7
    atr_loss_margin: float = 1.5
    hard_cutoff: float = 0.95
    bayes_windows: int = 30
    lower_sample: int = 7
    upper_sample: int = 7
    interval: str = "1min"
    fetch_huobi: bool = False
    simulate: bool = False

    def __post_init__(self):
        self.ATR_sample = int(self.ATR_sample)
        self.bayes_windows = int(self.bayes_windows)
        self.lower_sample = int(self.lower_sample)
        self.upper_sample = int(self.upper_sample)

