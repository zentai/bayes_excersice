import functools
from bayes_opt import BayesianOptimization
import pandas as pd

from config import config
from .strategy.turtle_trading import TurtleScout
from .engine.probabilistic_engine import BayesianEngine
from .hunterverse.interface import StrategyParam
from .hunterverse.interface import Symbol
from .sensor.market_sensor import LocalMarketSensor
from .sensor.market_sensor import HuobiMarketSensor
from .tradingfirm.pubsub_trader import xHunter
from .cloud_story import HuntingStory, DUMP_COL
from pydispatch import dispatcher
from config import config

DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir


def optimize_func(**kwargs):
    params = {
        # Period
        "interval": "1day",
        "funds": 100,
        "stake_cap": 100,
        "symbol": Symbol("btcusdt"),
        "backtest": True,
    }
    params.update(kwargs)
    sp = StrategyParam(**params)

    base_df = None
    if sp.backtest:
        sensor = LocalMarketSensor(symbol=sp.symbol, interval=sp.interval)
        # sensor = MongoMarketSensor(symbol=sp.symbol, interval=sp.interval)
    else:
        sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)
    scout = TurtleScout(params=sp)
    engine = BayesianEngine(params=sp)
    hunter = xHunter(params=sp)
    base_df = sensor.scan(2000 if not sp.backtest else 100)
    story = HuntingStory(sensor, scout, engine, hunter, base_df)
    dispatcher.connect(story.move_forward, signal="k_channel")
    dispatcher.connect(story.sim_attack_feedback, signal="sim_attack_feedback")
    pub_thread = story.pub_market_sensor(sp)
    pub_thread.start()
    pub_thread.join()
    # print(story.base_df[DUMP_COL])

    # sensor.db.save(collection_name=f"{sp.symbol.name}_review", df=review)
    story.base_df[DUMP_COL].to_csv(f"{REPORTS_DIR}/{sp}.csv", index=False)
    # print(f"created: {REPORTS_DIR}/{sp}.csv")
    review = story.hunter.review_mission(story.base_df)
    return review.Profit


def run():

    params = {
        "ATR_sample": (3, 30),
        "bayes_windows": (3, 30),
        "lower_sample": (3, 30),
        "upper_sample": (3, 30),
        "hard_cutoff": (0.5, 0.99),
        "profit_loss_ratio": (0.5, 10),
        "atr_loss_margin": (1, 10),
        "surfing_level": (1, 10),
    }

    optimizer = BayesianOptimization(
        f=optimize_func,
        pbounds=params,
        random_state=1,
        allow_duplicate_points=True,
    )
    optimizer.maximize(init_points=5, n_iter=100)
    best_params = optimizer.max["params"]
    best_score = optimizer.max["target"]
    import pprint

    print(f"    best_params = {pprint.pformat(best_params)}")
    print("best_score = ", best_score)
    best_params["Name"] = params.symbol.name
    best_params["best_score"] = best_score
    best_params["debug"] = False

    print(e)


# scanning_df.to_csv(f"{REPORTS_DIR}/scanning_{len(test_cases)}.csv", index=False)
# print(f"created: {REPORTS_DIR}/scanning_{len(test_cases)}.csv")


if __name__ == "__main__":
    run()
