# 文件路径：quanthall/dispatcher.py

import json
import time
from pathlib import Path
from datetime import datetime
import shutil
from quanthall.front_mission import FrontMission
from quanthall.quant_hunt_bank import QuantHuntBank
from quanthunt.execute import parse_strategy_from_task_id
from quanthunt.config.core_config import config
from quanthunt.hunterverse.interface import StrategyParam, Symbol
from quanthunt.utils.telegram_helper import telegram_msg


class QuantHall:
    def __init__(self):
        self.bank = QuantHuntBank(total_funds=47)
        self.status_dir = Path(f"{config.data_dir}/status")
        self.archive_dir = Path(f"{config.data_dir}/archive")
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def restart_task(self, task_id: str):
        print(f"[QuantHall] Restarting task {task_id}...")
        strategy_dict = parse_strategy_from_task_id(task_id)
        sp = StrategyParam(**strategy_dict)
        fm = FrontMission(sp)
        fm.start()

    def scan_status(self, timeout_sec: int = 600) -> set:
        active_mission = set()
        for file in self.status_dir.glob("*.json"):
            with open(file) as f:
                status = json.load(f)
            task_id = status.get("task_id")
            state = status.get("status")
            result = status.get("result", {})
            symbol = result.get("symbol") or parse_strategy_from_task_id(task_id).get(
                "symbol"
            )
            profit = result.get("Profit") or 0
            if state == "Completed":
                message = f"✅ Task completed\nSymbol: {symbol}\nProfit: {profit:.4f}"
                telegram_msg(message)
                self.bank.release(f"{task_id}", profit)
                shutil.move(str(file), self.archive_dir / file.name)
                continue

            last_update = datetime.strptime(
                status.get("last_update"), "%Y-%m-%dT%H:%M:%S"
            )
            seconds_since = (datetime.now() - last_update).total_seconds()

            if state == "Running" and seconds_since > timeout_sec:
                print(
                    f"[QuantHall] Task {task_id} unresponsive for {seconds_since} sec. Restarting..."
                )
                self.restart_task(task_id)
            elif state == "Running":
                active_mission.add((symbol, profit))
        return active_mission

    def can_launch(self, strategy: StrategyParam, active_symbols: set) -> bool:
        if strategy.symbol.name in active_symbols:
            print(
                f"[QuantHall] Task for {strategy.symbol.name} already running. Skipping."
            )
            return False
        # if not self.bank.get_balance() >= strategy.funds:
        #     print(f"[QuantHall] Not enough funds to launch {strategy.symbol.name}")
        #     return False
        return True

    def launch_task(self, strategy: StrategyParam):
        self.bank.allocate(f"{strategy}", strategy.funds)
        mission = FrontMission(strategy)
        pid = mission.start()
        print(f"[QuantHall] Launched new task {strategy.task_id} with PID {pid}")

    def discover_and_dispatch(self, symbol, interval, funds, stake_cap):
        active_mission = self.scan_status()
        print(f"[QuantHall] Active symbols: {active_mission}")
        active_symbols = set([mission[0] for mission in active_mission])

        strategy = StrategyParam(
            symbol=Symbol(symbol),
            interval=interval,
            funds=funds,
            stake_cap=stake_cap,
            hmm_split=3,
            api_key="TBD",
            secret_key="TBD",
        )
        if self.can_launch(strategy, active_symbols):
            self.launch_task(strategy)

    def run_once(self, symbol, interval, funds, stake_cap):
        print("[QuantHall] Running one-time scan...")
        self.discover_and_dispatch(symbol, interval, funds, stake_cap)

    def run_loop(self, interval_sec: int = 3600):
        while True:
            self.run_once()
            print(f"[QuantHall] Sleeping {interval_sec} sec...")
            time.sleep(interval_sec)


if __name__ == "__main__":
    hall = QuantHall()
    hall.run_once()
