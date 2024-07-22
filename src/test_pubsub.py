from .hunterverse.interface import StrategyParam
from .tradingfirm.pubsub_trader import xHunter
from .hunterverse.interface import Symbol

if __name__ == "__main__":
    

    params = {
        "ATR_sample": 60,
        "atr_loss_margin": 3.5,
        "hard_cutoff": 0.95,
        "profit_loss_ratio": 3.0,
        "bayes_windows": 10,
        "lower_sample": 30.0,
        "upper_sample": 30.0,
        "interval": "1day",
        "funds": 100,
        "stake_cap": 10.5,
        "symbol": None,
        "surfing_level": 6,
        "fetch_huobi": True,
        "simulate": True,
    }
    params.update(
        {
            "interval": "1day",
            "funds": 100,
            "stake_cap": 50,
            "symbol": Symbol("btcusdt"),
        }
    )
    sp = StrategyParam(**params)
    hunter = xHunter(params=sp)
    hunter.sim_attack(hunting_id='2024-07-20', target_price=100, trigger_price=99, order_type='b', kelly=1, market_High=100, market_Low=99)
