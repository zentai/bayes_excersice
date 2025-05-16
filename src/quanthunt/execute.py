import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import click
import os
from dotenv import load_dotenv
from quanthunt.hunterverse.interface import StrategyParam
from quanthunt.story import start_journey
from quanthunt.hunterverse.interface import Symbol
from quanthunt.utils import pandas_util

# Load API credentials from .env file
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
api_key = os.getenv("API_KEY")
secret_key = os.getenv("SECRET_KEY")


def parse_strategy_from_task_id(task_id: str) -> dict:
    try:
        parts = task_id.split("_")
        task_id = "_".join(parts[:2])
        symbol = parts[2]
        interval = parts[3]

        joined = "_".join(parts[4:])  # join param string

        def extract(pattern):
            match = re.search(pattern, joined)
            return match.group(1) if match else None

        return {
            "task_id": task_id,
            "symbol": Symbol(symbol),
            "interval": interval,
            "funds": float(extract(r"fun([\d\.]+)")),
            "stake_cap": float(extract(r"cap([\d\.]+)")),
            "ATR_sample": int(extract(r"atr(\d+)")),
            "bayes_windows": int(extract(r"bw(\d+)")),
            "upper_sample": int(extract(r"up(\d+)")),
            "lower_sample": int(extract(r"lw(\d+)")),
            "hmm_split": int(extract(r"hmm(\d+)")),
            "hard_cutoff": float(extract(r"cut([\d\.]+)")),
            "profit_loss_ratio": float(extract(r"pnl([\d\.]+)")),
            "atr_loss_margin": float(extract(r"ext([\d\.]+)")),
            "surfing_level": int(extract(r"stp(\d+)")),
            "api_key": api_key,
            "secret_key": secret_key,
        }
    except Exception as e:
        raise ValueError(f"Failed to parse task_id: {task_id}") from e


@click.command()
@click.option("--dispatch", type=str, help="dispatch params hash for hunter")
def cli_main(dispatch):
    overrides = parse_strategy_from_task_id(dispatch)
    sp = pandas_util.build_strategy_param(overrides)
    start_journey(sp)


if __name__ == "__main__":
    cli_main()
