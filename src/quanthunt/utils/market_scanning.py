import time
from datetime import datetime
from typing import Dict, List
from statistics import mean, stdev
from typing import Dict, Any

import pandas as pd
import click
from huobi.client.market import MarketClient
from huobi.constant import CandlestickInterval
from quanthunt.hunterverse.interface import DEBUG_COL
from quanthunt.utils import pandas_util
from quanthunt.utils.telegram_helper import telegram_msg
from quanthunt.tradingfirm.platforms.huobi_api import Candlestick
import traceback

import os
from quanthunt.sensor.market_sensor import HuobiMarketSensor
from quanthunt.hunterverse.storage import HuntingCamp
from quanthunt.hunterverse.interface import StrategyParam
from quanthunt.hunterverse.interface import Symbol
from quanthunt.strategy.turtle_trading import TurtleScout, emv_cross_strategy

INTERVAL_MAP: Dict[str, CandlestickInterval] = {
    "1min": CandlestickInterval.MIN1,
    "5min": CandlestickInterval.MIN5,
    "15min": CandlestickInterval.MIN15,
    "30min": CandlestickInterval.MIN30,
    "60min": CandlestickInterval.MIN60,
}


class CoinAnalyzer:
    def __init__(self, symbol: str, candles: list, period_sec: int):
        self.symbol = symbol
        self.period_sec = period_sec
        self.df = self._prepare_df(candles)

        self.filter_rules = {
            "drop_pct": 0.5,
            "HMM_min": 1,
            "Slope_max": 0.0001,
            "avg_hz_min": 0.4,
        }
        self.score_weight = {
            "drop_pct": 0.5,
            "HMM": 0.2,
            "avg_hz": 0.1,
            "Slope": 0.2,
        }
        self.result = self.analyze()
        self.result["Score"] = self.score()

    def _prepare_df(self, candles):
        df = pd.DataFrame(
            [
                Candlestick(
                    stick.id,
                    stick.high,
                    stick.low,
                    stick.open,
                    stick.close,
                    stick.amount,
                    stick.count,
                    stick.vol,
                )
                for stick in candles
            ]
        )
        overrides = {
            "funds": 50,
            "stake_cap": 10,
            "hmm_split": 4,
            "bayes_windows": 20,
            "symbol": Symbol(self.symbol),
            "backtest": True,
            "debug_mode": ["statement"],
            "api_key": os.getenv("API_KEY"),
            "secret_key": os.getenv("SECRET_KEY"),
        }
        sp = pandas_util.build_strategy_param(overrides)
        sensor = HuobiMarketSensor(symbol=sp.symbol, interval=sp.interval)
        camp = HuntingCamp(sp, sensor)
        scout = TurtleScout(params=sp, buy_signal_func=emv_cross_strategy)
        scout = TurtleScout(sp)
        df = camp.update()

        df["hz"] = df["Count"] / self.period_sec
        df["pct_change"] = df["Close"].pct_change().fillna(0)
        df = scout.train(df)
        return df

    def _hmm_score(self):
        s = (self.df["HMM_Signal"] == 1).astype(int)  # 1 for up-state, 0 otherwise
        streak = s.iloc[::-1].cumprod().sum()  # reverse → cumprod → sum
        if streak > 3:
            print(f"Symbol: {self.symbol} [{streak}]")
            print(self.df[DEBUG_COL][-15:])
        return int(streak)

    def _slope(self):
        return self.df.Slope.iloc[0]

    def _drop_pct(self):
        drop_pct = (
            (self.df["Close"].iloc[-1] - self.df["Close"].iloc[0])
            / self.df["Close"].iloc[0]
            if len(self.df) > 1
            else 0.0
        )
        avg_hz = mean(self.df["hz"])
        std_hz = self.df["hz"].std()
        max_hz = self.df["hz"].max()
        min_hz = self.df["hz"].min()
        drop_from_high = (
            (self.df["Close"].iloc[-1] - self.df["High"].max()) / self.df["High"].max()
            if len(self.df) > 0
            else 0.0
        )
        running_max = self.df["Close"].cummax()
        drawdowns = (self.df["Close"] - running_max) / running_max
        max_drawdown = drawdowns.min() if not drawdowns.empty else 0.0
        print(self.symbol, drop_from_high, max_drawdown)
        return drop_pct, avg_hz, std_hz, max_hz, min_hz, drop_from_high, max_drawdown

    def analyze(self) -> Dict[str, Any]:
        drop_pct, avg_hz, std_hz, max_hz, min_hz, drop_from_high, max_drawdown = (
            self._drop_pct()
        )

        self.result = {
            "symbol": self.symbol,
            "drop_pct": drop_pct,
            "drop_from_high": drop_from_high,
            "max_drawdown": max_drawdown,
            "avg_hz": avg_hz,
            "std_hz": std_hz,
            "max_hz": max_hz,
            "min_hz": min_hz,
            "valid_points": len(self.df),
            "HMM": self._hmm_score(),
            "Slope": self._slope(),
            "link": f"https://www.htx.com/trade/{self.symbol.replace('usdt', '_usdt')}/",
        }
        return self.result

    def is_candidate(self) -> bool:
        r = self.result
        return all(
            [
                # r["drop_pct"] <= self.filter_rules.get("drop_pct_max", 0),
                r["HMM"] >= self.filter_rules.get("HMM_min", 0),
                # r["drop_from_high"] <= -0.1,
                # r["max_drawdown"] >= 1,
                # r["Slope"] <= self.filter_rules.get("Slope_max", 0),
                # r["avg_hz"] >= self.filter_rules.get("avg_hz_min", 0),
            ]
        )

    def score(self) -> float:
        r = self.result
        return (
            r["drop_pct"] * self.score_weight.get("drop_pct", 0.0)
            + r["drop_from_high"] * self.score_weight.get("drop_from_high", 0.0)
            + r["max_drawdown"] * self.score_weight.get("max_drawdown", 0.0)
            + r["HMM"] * self.score_weight.get("HMM", 0.0)
            + r["avg_hz"] * self.score_weight.get("avg_hz", 0.0)
            + r["Slope"] * self.score_weight.get("Slope", 0.0)
        )


def get_active_symbols(mc: MarketClient, volume_threshold: float) -> List[str]:
    tickers = mc.get_market_tickers()
    symbols = [
        t.symbol
        for t in tickers
        if (t.symbol.endswith("usdt") and t.amount >= volume_threshold)
    ]
    click.echo(
        f"[✓] 获取 {len(tickers)} 个 ticker，筛选出 {len(symbols)} 个 USDT 交易对成交量 > {volume_threshold}"
    )
    return symbols


def calc_avg_hz(
    mc: MarketClient,
    symbol: str,
    interval_const: CandlestickInterval,
    size: int,
    period_sec: int,
) -> Dict[str, Any]:
    try:
        candles = mc.get_candlestick(symbol, interval_const, size)
        analyzer = CoinAnalyzer(
            symbol=symbol,
            candles=candles,
            period_sec=period_sec,
        )
        return analyzer
    except Exception as e:
        print(f"[ERROR] 拉取 {symbol} 出错: {e}")
        # print(traceback.format_exc())
        return None


def run_once(
    interval: str, hours: int, volume_threshold: float, hz_threshold: float
) -> None:
    mc = MarketClient(init_log=False)
    interval_key = interval.lower()
    if interval_key not in INTERVAL_MAP:
        raise click.BadParameter(f"不支持的 interval: {interval}")

    interval_const = INTERVAL_MAP[interval_key]
    period_minutes = int(interval_key.replace("min", "").replace("hour", "60"))
    size = int(hours * 60 / period_minutes)
    period_sec = period_minutes * 60

    click.echo(
        f"[→] 参数：interval={interval} | hours={hours} | size={size} | 每段={period_sec}s"
    )

    symbols = get_active_symbols(mc, volume_threshold)

    rows = []
    for i, sym in enumerate(symbols):
        click.echo(f"[...] 正在处理 {i+1}/{len(symbols)}: {sym}")
        coin = calc_avg_hz(mc, sym, interval_const, size, period_sec)
        if coin and coin.is_candidate():
            rows.append(coin.result)

    if not rows:
        click.echo("[⚠️] 无符合条件的交易对")
        return

    df = (
        pd.DataFrame(rows)
        .sort_values(["drop_from_high", "HMM", "Slope"], ascending=False)
        .reset_index(drop=True)
    )
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    click.echo(f"\n[✓] {timestamp} 结果共 {len(df)} 项，按 Hz 降序排序：")
    click.echo(
        df[
            [
                "symbol",
                "HMM",
                "avg_hz",
                "std_hz",
                "Slope",
                "drop_from_high",
                "Score",
                "link",
            ]
        ].to_string(
            index=False,
            formatters={"avg_hz": "{:.2f}".format, "std_hz": "{:.2f}".format},
        )
    )
    # telegram_msg(
    #     f"""<pre>{df[["symbol", "avg_hz", "std_hz"]].to_markdown(index=False)}</pre>"""
    # )


@click.command()
@click.option(
    "--interval",
    default="5min",
    show_default=True,
    help="K线周期，例如 1min / 5min / 60min",
)
@click.option("--hours", default=1, show_default=True, help="向后取多少小时的数据")
@click.option(
    "--volume-threshold",
    default=1_000_000,
    show_default=True,
    help="24h USDT 成交量阈值",
)
@click.option(
    "--hz-threshold", default=1.0, show_default=True, help="平均成交频率阈值 (Hz)"
)
@click.option("--loop", is_flag=True, help="是否持续轮询执行")
@click.option("--sleep-sec", default=3600, show_default=True, help="轮询模式间隔秒数")
def main(interval, hours, volume_threshold, hz_threshold, loop, sleep_sec):
    if loop:
        while True:
            try:
                run_once(interval, hours, volume_threshold, hz_threshold)
            except Exception as e:
                click.echo(f"[ERROR] {e}")
            time.sleep(max(1, sleep_sec))
    else:
        run_once(interval, hours, volume_threshold, hz_threshold)


if __name__ == "__main__":
    main()
