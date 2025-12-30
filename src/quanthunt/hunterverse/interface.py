from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

ZERO = 1e-8

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

    def __format__(self, format_spec):
        if format_spec == "price":
            return f"{self.name} price precision: {self.price_prec}"
        elif format_spec == "amount":
            return f"{self.name} amount precision: {self.amount_prec}"
        else:
            return self.name


@dataclass
class StrategyParam:
    api_key: str
    secret_key: str
    symbol: str
    ATR_sample: int = 7
    atr_loss_margin: float = 1.5
    hard_cutoff: float = 0.95
    profit_loss_ratio: float = 2.0
    bayes_windows: int = 30
    lower_sample: int = 7
    upper_sample: int = 7
    surfing_level: int = 5
    funds: float = 100
    stake_cap: int = 50
    interval: str = "1min"
    hmm_split: int = 5
    hmm_model: str = "trend"
    backtest: bool = False
    debug_mode: list = field(default_factory=list)
    load_deals: list = field(default_factory=list)
    start_deal: int = 0
    task_id: Optional[str] = None

    def __post_init__(self):
        self.ATR_sample = int(self.ATR_sample)
        self.bayes_windows = int(self.bayes_windows)
        self.lower_sample = int(self.lower_sample)
        self.upper_sample = int(self.upper_sample)
        self.hmm_split = int(self.hmm_split)
        if self.task_id is None:
            self.task_id = datetime.now().strftime("%m%d_%H%M%S")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        # btcusdt1min_atr15bw15up15lw15_cut0.99pnl2ext3stp3
        header = f"{self.symbol.name}_{self.interval}"
        buy_params = f"fun{self.funds}cap{self.stake_cap}atr{self.ATR_sample}bw{self.bayes_windows}up{self.upper_sample}lw{self.lower_sample}hmm{self.hmm_split}"
        sell_params = f"cut{self.hard_cutoff}pnl{self.profit_loss_ratio}ext{self.atr_loss_margin}stp{self.surfing_level}"
        return f"{self.task_id}_{header}_{buy_params}_{sell_params}"


@dataclass
class xOrder:
    order_id: str


@dataclass
class xBuyOrder(xOrder):
    target_price: float
    kelly: float
    executed_price: Optional[float] = None
    operator: str = ""
    order_type: str = "B"
    status: str = "unfilled"
    timestamp: datetime = field(default_factory=datetime.now)
    client: str = ""
    position: float = 0


@dataclass
class xSellOrder(xOrder):
    atr_exit_price: float
    profit_leave_price: float
    executed_price: Optional[float] = None
    operator: str = ""
    order_type: str = "S"
    status: str = "unfilled"
    timestamp: datetime = field(default_factory=datetime.now)
    client: str = ""
    position: float = 0
    cutoff_price: float = 0


DEBUG_COL = [
    "Date",
    # "Open",
    # "High",
    # "Low",
    "Close",
    # "Count_Hz",
    # "turtle_h",
    # "ema_short",
    # "ema_long",
    "BuySignal",
    # "Stop_profit",
    "exit_price",
    # "time_cost",
    # "buy",
    # "sell",
    "profit",
    # "OBV",
    # "OBV_UP",
    # "Matured",
    # "Kelly",
    # "Postrior",
    # "P/L",
    # "likelihood",
    # "profit_margin",
    # "loss_margin",
    # "Kalman",
    # "HMM_State",
    "HMM_Epoch",
    "HMM_Signal",
    # MOSAIC model
    "m_pc",
    # "m_pt_speed",
    # "m_pt_accel",
    "m_force",
    # "m_force_trend",
    # "m_force_bias",
    # "m_z_price",
    # "m_z_force",
    # "m_z_mix",
    "m_regime_noise_level",
    "c_center",
    # "c_z_center",
    # BOCPD
    "bocpd_phase",
    "bocpd_cp_prob",
    # "bocpd_runlen_mean",
    "bocpd_runlen_mode",
    # "bocpd_risk",
    # "bocpd_tail",
    # "bocpd_shock",
]

DUMP_COL = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "HMM_State",
    "HMM_Epoch",
    "HMM_Signal",
    "Kalman",
    # "Count",
    "Vol",
    "turtle_h",
    "BuySignal",
    "ema_short",
    "ema_long",
    "Stop_profit",
    "exit_price",
    "Matured",
    "time_cost",
    "buy",
    "sell",
    "profit",
    "P/L",
    # "turtle_l",
    # "turtle_h",
    # "Kelly",
    # "Postrior",
    # "likelihood",
    # "profit_margin",
    # "loss_margin",
    # "drift",
    # "volatility",
    # MOSAIC model
    "m_pc",
    "m_pt_speed",
    "m_pt_accel",
    "m_force",
    "m_force_trend",
    "m_force_bias",
    "m_z_price",
    "m_z_force",
    "m_z_mix",
    "m_regime_noise_level",
    "c_center",
    "c_z_center",
    # BOCPD
    "bocpd_phase",
    "bocpd_cp_prob",
    "bocpd_runlen_mean",
    "bocpd_runlen_mode",
    "bocpd_risk",
    "bocpd_tail",
    "bocpd_shock",
]
