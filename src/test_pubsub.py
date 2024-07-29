from .hunterverse.interface import StrategyParam
from .tradingfirm.pubsub_trader import xHunter
from .hunterverse.interface import Symbol
from pydispatch import dispatcher
from .sensor.market_sensor import LocalMarketSensor

def sim_attack_feedback(order_id, order_status, price, position):
        print(f"Please update dataframe here: =====>>  {order_id, order_status, price, position=}")
        # base_df.loc[s_buy_order, "sBuy"] = target_price
        # base_df.loc[s_buy_order, "sPosition"] = self.sim_bag.position
        # # base_df.loc[s_buy_order, "sCash"] = self.sim_bag.cash
        # base_df.loc[s_buy_order, "sAvgCost"] = self.sim_bag.avg_cost


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

    dispatcher.connect(sim_attack_feedback, signal="sim_attack_feedback")
    sp = StrategyParam(**params)

    sensor = LocalMarketSensor(symbol=sp.symbol, interval="local")
    hunter = xHunter(params=sp)

    base_df = sensor.scan(100)
    print(base_df)
    round = sensor.left() or 1000000
    for i in range(round):
        base_df = sensor.fetch(base_df)
        date, open, high, low, close, vol = base_df.iloc[-1]
        hunter.sim_attack(hunting_id=date, target_price=close, order_type='b', kelly=1, market_High=high, market_Low=low)
