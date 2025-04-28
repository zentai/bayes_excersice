import time
from datetime import datetime
from typing import Dict, List
from statistics import mean, stdev
from typing import Dict, Any

import pandas as pd
import click
from huobi.client.market import MarketClient
from huobi.constant import CandlestickInterval
from quanthunt.utils.telegram_helper import telegram_msg
import traceback

INTERVAL_MAP: Dict[str, CandlestickInterval] = {
    "1min": CandlestickInterval.MIN1,
    "5min": CandlestickInterval.MIN5,
    "15min": CandlestickInterval.MIN15,
    "30min": CandlestickInterval.MIN30,
    "60min": CandlestickInterval.MIN60,
}


def get_active_symbols(mc: MarketClient, volume_threshold: float) -> List[str]:
    tickers = mc.get_market_tickers()
    symbols = [
        t.symbol
        for t in tickers
        if t.symbol.endswith("usdt") and t.amount >= volume_threshold
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
    debug: bool = False,
) -> Dict[str, Any]:
    try:
        candles = mc.get_candlestick(symbol, interval_const, size)
        hz_list = [c.count / period_sec for c in candles if c and c.count is not None]
        import os
        from quanthunt.hunterverse.interface import StrategyParam
        from quanthunt.hunterverse.interface import Symbol
        from quanthunt.strategy.turtle_trading import TurtleScout, emv_cross_strategy
        from quanthunt.tradingfirm.platforms.huobi_api import Candlestick

        candlesticks = [
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
        base_df = pd.DataFrame(candlesticks)

        params = {
            "ATR_sample": 60,
            "bayes_windows": 10,
            "lower_sample": 60,
            "upper_sample": 60,
            "hard_cutoff": 0.9,
            "profit_loss_ratio": 3,
            "atr_loss_margin": 1.5,
            "surfing_level": 5,
            "interval": "5min",
            "funds": 50,
            "stake_cap": 10,
            "symbol": Symbol(symbol),
            "backtest": True,
            "debug_mode": ["statement"],
            "api_key": os.getenv("API_KEY"),
            "secret_key": os.getenv("SECRET_KEY"),
        }
        sp = StrategyParam(**params)
        scout = TurtleScout(params=sp, buy_signal_func=emv_cross_strategy)
        scout = TurtleScout(sp)
        base_df = scout.train(base_df)
        if not hz_list:
            return {
                "symbol": symbol,
                "HMM": 0,
                "avg_hz": 0.0,
                "std_hz": 0.0,
                "max_hz": 0.0,
                "min_hz": 0.0,
                "hz_list": [],
                "valid_points": 0,
            }

        avg_hz = mean(hz_list)
        std_hz = stdev(hz_list) if len(hz_list) > 1 else 0.0
        result = {
            "symbol": symbol,
            "HMM": base_df.UP_State.iloc[-1],
            "avg_hz": avg_hz,
            "std_hz": std_hz,
            "max_hz": max(hz_list),
            "min_hz": min(hz_list),
            "hz_list": hz_list,
            "valid_points": len(hz_list),
            "link": f"https://www.htx.com/trade/{symbol.replace('usdt', '_usdt')}/",
        }

        if debug:
            print(f"[debug] {symbol} Hz list: {hz_list}")
            print(f"[debug] {symbol} avg={avg_hz:.2f} Hz, std={std_hz:.2f}")

        return result
    except Exception as e:
        print(f"[ERROR] 拉取 {symbol} 出错: {e}")
        # print(traceback.format_exc())
        return {
            "symbol": symbol,
            "HMM": 0,
            "avg_hz": 0.0,
            "std_hz": 0.0,
            "max_hz": 0.0,
            "min_hz": 0.0,
            "hz_list": [],
            "valid_points": 0,
        }


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
        res = calc_avg_hz(mc, sym, interval_const, size, period_sec)
        if res["avg_hz"] >= hz_threshold and res["HMM"] == 1:
            rows.append(res)

    if not rows:
        click.echo("[⚠️] 无符合条件的交易对")
        return

    df = (
        pd.DataFrame(rows).sort_values("std_hz", ascending=False).reset_index(drop=True)
    )
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    click.echo(f"\n[✓] {timestamp} 结果共 {len(df)} 项，按 Hz 降序排序：")
    click.echo(
        df[["symbol", "avg_hz", "std_hz", "link"]].to_string(
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
