from settings import DATA_DIR, SRC_DIR, REPORTS_DIR
import pandas as pd


def load_symbols(symbols):
    code = symbols  # "BTC-USD"
    df = pd.read_csv(f"{DATA_DIR}/{code}.csv")
    df = df.dropna()
    size = len(df)

    start_idx = 0
    windows = 200
    # TODO: Consider creating a function to handle the creation of the base dataframe
    return df[:1024].copy()
