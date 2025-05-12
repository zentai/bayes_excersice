import pandas as pd
from datetime import datetime
from pathlib import Path
from quanthunt.config.core_config import config


class QuantHuntBank:
    def __init__(
        self,
        total_funds: float,
        history_path: str = f"{config.data_dir}/fund_history.csv",
    ):
        self.total_funds = total_funds
        self.available_funds = total_funds
        self.allocated = {}  # task_id -> amount
        self.history = pd.DataFrame(
            columns=["time", "action", "task_id", "amount", "pnl", "balance"]
        )
        self.history_path = Path(history_path)
        self._load_history()

    def _log(self, action: str, task_id: str, amount: float, pnl: float = None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "time": now,
            "action": action,
            "task_id": task_id,
            "amount": amount,
            "pnl": pnl,
            "balance": self.available_funds,
        }
        self.history.loc[len(self.history)] = record

    def _load_history(self):
        if self.history_path.exists():
            self.history = pd.read_csv(self.history_path)
            if not self.history.empty:
                self.available_funds = float(self.history.iloc[-1]["balance"])

    def allocate(self, task_id: str, amount: float) -> bool:
        if self.available_funds >= amount:
            self.available_funds -= amount
            self.allocated[task_id] = amount
            self._log("allocate", task_id, amount)
            return True
        return False

    def release(self, task_id: str, pnl: float = 0.0):
        if task_id in self.allocated:
            original = self.allocated.pop(task_id)
            self.available_funds += original + pnl
            self._log("release", task_id, original, pnl)

    def get_balance(self) -> float:
        return self.available_funds

    def get_allocation(self) -> dict:
        return self.allocated.copy()

    def save_history(self):
        self.history.to_csv(self.history_path, index=False)

    def get_balance_curve(self):
        releases = self.history[self.history["action"] == "release"]
        # return releases[["time", "balance"]].copy()
        return releases


if __name__ == "__main__":
    bank = QuantHuntBank(
        total_funds=20.0, history_path=f"{config.data_dir}/fund_history_unit-test.csv"
    )

    print("初始余额:", bank.get_balance())

    # Task 1
    bank.allocate("task1", 5.0)
    print("[Task1] 分配后余额:", bank.get_balance())

    print("当前分配记录:")
    print(bank.get_allocation())

    # Task 2
    bank.allocate("task2", 7.0)
    print("[Task2] 分配后余额:", bank.get_balance())

    print("当前分配记录:")
    print(bank.get_allocation())

    # Release Task 1
    bank.release("task1", pnl=1.2)
    print("[Task1] 释放后余额:", bank.get_balance())

    print("当前分配记录:")
    print(bank.get_allocation())

    # Release Task 2
    bank.release("task2", pnl=0.5)
    print("[Task2] 释放后余额:", bank.get_balance())

    print("当前分配记录:")
    print(bank.get_allocation())

    print("资金曲线:")
    print(bank.get_balance_curve())

    bank.save_history()
