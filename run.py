import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import click
import os
from dotenv import load_dotenv
from quanthunt.hunterverse.interface import StrategyParam
from quanthunt.story import start_journey
from quanthunt.hunterverse.interface import Symbol

# Load API credentials from .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
api_key = os.getenv("API_KEY")
secret_key = os.getenv("SECRET_KEY")

# very good for BTCUSDT day K
# params = {
#     "ATR_sample": 15,
#     "atr_loss_margin": 3,
#     "hard_cutoff": 0.9,
#     "profit_loss_ratio": 2,
#     "bayes_windows": 15,
#     "lower_sample": 15.0,
#     "upper_sample": 15.0,
#     "interval": "1min",
#     "funds": 100,
#     "stake_cap": 50,
#     "symbol": None,
#     "surfing_level": 3,
#     "fetch_huobi": False,
#     "simulate": True,
# }


@click.command()
@click.option("--symbol", default="moveusdt", help="Trading symbol (e.g. trxusdt)")
@click.option("--interval", default="1min", help="Trading interval")
@click.option("--funds", default=50.2, type=float, help="Available funds")
@click.option("--cap", default=10.1, type=float, help="Stake cap")
@click.option("--deals", default="", help="Comma separated deal IDs")
@click.option("--start_deal", default=0, type=int, help="Start to load from deal ID")
@click.option("--hmm_split", default=5, type=int, help="hmm status split")
def cli_main(symbol, interval, funds, cap, deals, start_deal, hmm_split):
    deal_ids = [int(x.strip()) for x in deals.split(",") if x.strip()] if deals else []

    params = {
        "ATR_sample": 60,
        "bayes_windows": 10,
        "lower_sample": 60,
        "upper_sample": 60,
        "hard_cutoff": 0.95,
        "profit_loss_ratio": 3,
        "atr_loss_margin": 3,
        "surfing_level": 2,
        "interval": interval,
        "funds": funds,
        "stake_cap": cap,
        "symbol": Symbol(symbol),
        "hmm_split": hmm_split,
        "backtest": False,
        "debug_mode": [
            "statement",
            "statement_to_csv",
            "mission_review",
            "final_statement_to_csv",
        ],
        "load_deals": deal_ids,
        "start_deal": start_deal,
        "api_key": api_key,
        "secret_key": secret_key,
    }
    sp = StrategyParam(**params)
    start_journey(sp)


if __name__ == "__main__":
    cli_main()
