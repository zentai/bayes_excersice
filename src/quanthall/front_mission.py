import subprocess
import json
from datetime import datetime
from pathlib import Path
from quanthunt.config.core_config import config
from quanthunt.hunterverse.interface import StrategyParam


class FrontMission:
    def __init__(self, strategy: StrategyParam):
        self.strategy = strategy
        self.task_id = self.strategy.generate_task_id()
        self.process = None
        self.status_path = Path(f"{config.data_dir}/status/{self.task_id}.json")

    def start(self):
        cmd = [
            "python",
            "quanthunt/run.py",
            "--symbol",
            self.strategy.symbol,
            "--interval",
            self.strategy.interval,
            "--funds",
            str(self.strategy.funds),
            "--cap",
            str(self.strategy.stake_cap),
            "--hmm_split",
            str(self.strategy.hmm_split),
        ]
        self.process = subprocess.Popen(cmd)
        self._write_initial_status(self.process.pid)
        return self.process.pid

    def _write_initial_status(self, pid: int):
        status_data = {
            "task_id": self.task_id,
            "status": "running",
            "last_update": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "pid": pid,
            "result": {},
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_path, "w") as f:
            json.dump(status_data, f, indent=2)

    def is_alive(self, timeout_sec: int = 60) -> bool:
        if not self.status_path.exists():
            return False
        with open(self.status_path) as f:
            status = json.load(f)
        last_update = datetime.strptime(status["last_update"], "%Y-%m-%dT%H:%M:%S")
        return (datetime.now() - last_update).total_seconds() < timeout_sec

    def is_finished(self) -> bool:
        if not self.status_path.exists():
            return False
        with open(self.status_path) as f:
            status = json.load(f)
        return status.get("status") == "finished" and not status.get("result", {}).get(
            "HOLD", True
        )

    def kill(self):
        if self.process:
            self.process.kill()


if __name__ == "__main__":
    from quanthunt.hunterverse.interface import Symbol

    strategy = StrategyParam(
        api_key="demo",
        secret_key="demo",
        symbol=Symbol("bbusdt"),
        interval="5min",
        funds=10.0,
        stake_cap=5.0,
        hmm_split=3,
    )
    mission = FrontMission(strategy)
    pid = mission.start()
    print(f"Started FrontMission with PID: {pid} | Task ID: {mission.task_id}")
