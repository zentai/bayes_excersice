import subprocess
import json
from datetime import datetime
from pathlib import Path
from quanthunt.config.core_config import config
from quanthunt.hunterverse.interface import StrategyParam


class FrontMission:
    def __init__(self, sp: StrategyParam):
        self.sp = sp
        self.process = None
        self.status_path = Path(f"{config.data_dir}/status/{self.sp}.json")

    def start(self):
        cmd = [
            "python",
            "-m",
            "quanthunt.execute",
            "--dispatch",
            str(self.sp),
        ]
        self.process = subprocess.Popen(cmd)
        self._write_initial_status(self.process.pid)
        return self.process.pid

    def _write_initial_status(self, pid: int):
        status_data = {
            "task_id": f"{self.sp}",
            "status": "Start",
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
        return status.get("status") == "Completed"

    def kill(self):
        if self.process:
            self.process.kill()


if __name__ == "__main__":
    from quanthunt.hunterverse.interface import Symbol

    task_id = datetime.now().strftime("%m%d_%H%M%S")
    strategy = StrategyParam(
        symbol=Symbol("donkeyusdt"),
        interval="5min",
        funds=15.0,
        stake_cap=15.0,
        hmm_split=5,
        task_id=task_id,
        api_key="TBD",
        secret_key="TBD",
    )
    mission = FrontMission(strategy)
    pid = mission.start()
    print(f"Started FrontMission with PID: {pid} | Task ID: {strategy}")

    import time

    while not mission.is_finished:
        print(f"Still Alive: {mission.is_alive()}")
        time.sleep(60 * 5)

    print(f"{strategy} completed!")
