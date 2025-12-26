import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


import logging
import os

import numpy as np
import pandas as pd
import statistics
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from quanthunt.strategy.algo_util.hmm_selector import HMMTrendSelector
from quanthunt.strategy.turtle_trading import (
    TurtleScout,
    buy_signal_from_mosaic_strategy,
)
from quanthunt.strategy.algo_util.performance import (
    compare_performance,
    compare_signal_filters,
    analyze_hmm_states,
    evaluate_hmm_signal,
    train_test_split_by_time,
)
from quanthunt.strategy.algo_util.kalman import (
    init_kalman_state,
    mosaic_step,
    build_kalman_params,
    prepare_mosaic_input,
    MosaicForceAdapter,
    MosaicPriceAdapter,
    CycleStateAdapter,
)
from quanthunt.strategy.algo_util.bocpdz import (
    BOCPDGaussianG0,
    BOCPDStudentTP1,
    DualBOCPD,
    PhaseFSMConfig,
    BOCPDPhaseFSM,
    DualBOCPDWrapper,
)
from quanthunt.hunterverse.interface import (
    IStrategyScout,
    IMarketSensor,
    IEngine,
    IHunter,
    StrategyParam,
    INTERVAL_TO_MIN,
    DEBUG_COL,
    DUMP_COL,
)
from quanthunt.hunterverse.interface import IStrategyScout, ZERO
from quanthunt.utils import pandas_util

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

from quanthunt.config.core_config import config

if __name__ == "__main__":
    params = {
        "ATR_sample": 60,
        "bayes_windows": 20,
        "lower_sample": 60,
        "upper_sample": 60,
        "hard_cutoff": 0.975,
        "profit_loss_ratio": 3,
        "atr_loss_margin": 1,
        "surfing_level": 5,
        "interval": "1min",
        "funds": 50,
        "stake_cap": 10,
        "hmm_split": 4,
        "backtest": True,
        "debug_mode": ["statement"],
        "api_key": "",
        "secret_key": "",
        "symbol": Symbol("btcusdt"),
    }

    sp = StrategyParam(**params)

    # ===== Data Loading =====
    base_df = pd.read_csv(f"/Users/Zen/Documents/code/bayes_excersice/data/btcusdt.csv")
    base_df = base_df[["Date", "Open", "High", "Low", "Close", "Vol"]]
    base_df["Date"] = pd.to_datetime(base_df["Date"])
    base_df = base_df.reset_index(drop=True)

    # ===== Split train & test =====
    top_10pct = int(len(base_df) * 0.5)
    train_df = base_df.iloc[:top_10pct].copy()
    test_df = base_df.iloc[top_10pct:].copy()
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    test_df["Matured"] = pd.NaT

    print(f"PROCESS table size: {len(base_df)}")

    # ===== Training Phase =====
    scout = TurtleScout(params=sp, buy_signal_func=buy_signal_from_mosaic_strategy)
    train_df = scout.train(train_df)
    # train_df = scout.market_recon(train_df)
    train_stats = compare_signal_filters(train_df)
    print(train_stats)
    print(analyze_hmm_states(train_df))

    if not os.path.exists(config.reports_dir):
        os.mkdir(config.reports_dir)

    update_idx = 0

    # ===== Step-by-step Online Backtest =====
    for _ in range(len(test_df)):
        new_row = test_df.iloc[update_idx].copy()
        new_row["Matured"] = pd.NaT
        new_row["Date"] = pd.to_datetime(new_row["Date"])

        new_df = pd.DataFrame([new_row], columns=train_df.columns)
        train_df = pd.concat([train_df, new_df], ignore_index=True)

        # 市場重建
        train_df = scout.market_recon(train_df)

        # Output in debug
        # print(_train[DUMP_COL])

        update_idx += 1

    _test_df = train_df.iloc[top_10pct:].copy()
    _test_df, _best = evaluate_hmm_signal(_test_df)
    _test_stats = compare_signal_filters(_test_df)
    print(_test_stats)

    print(analyze_hmm_states(_test_df))
