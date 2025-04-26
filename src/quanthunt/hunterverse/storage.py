import pandas as pd
from quanthunt.config.core_config import config


class HuntingCamp:
    """
    HuntingCamp is responsible for managing the local persistence of trading session data.
    It acts as a bridge between external market sensors and the internal historical record.

    Core responsibilities:
    - Load previous trading records from disk if available.
    - Fetch updated market data using a provided Sensor (Local or Online).
    - Merge existing records with newly scanned data.
    - Save the merged result back to disk.

    Note:
    HuntingCamp does NOT care about where the Sensor gets data (local file, online API, etc.).
    It only focuses on ensuring that the working DataFrame is up-to-date.
    """

    def __init__(self, strategy_param, sensor):
        self.strategy_param = strategy_param
        self.sensor = sensor
        self.file_path = config.reports_dir / f"{strategy_param}.csv"

    def load(self):
        """Load the saved trading DataFrame from disk, if it exists."""
        if self.file_path.exists():
            print(f"Loading from {self.file_path}")
            return pd.read_csv(self.file_path)
        else:
            print("No existing DataFrame found, starting new.")
            return pd.DataFrame()

    def update(self):
        """
        Update the current DataFrame:
        - First load existing records if any.
        - Scan new data from the Sensor.
        - Merge new data, avoiding duplicate dates.
        - Return the merged DataFrame.
        """
        base_df = self.load()

        # Always scan new market data
        update_df = self.sensor.scan(2000 if not self.strategy_param.backtest else 100)

        if not base_df.empty:
            update_df = update_df[~update_df["Date"].isin(base_df["Date"])]

        if not update_df.empty:
            base_df = pd.concat([base_df, update_df], ignore_index=True)

        return base_df

    def save(self, df):
        """Save the current DataFrame to disk."""
        df.to_csv(self.file_path, index=False)
        print(f"Saved DataFrame to {self.file_path}")
