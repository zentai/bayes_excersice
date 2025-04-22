import multiprocessing
import time
import yfinance as yf
from .utils.cloud import publish, subscribe
import pandas as pd

# 定义全局DataFrame
apple_df = pd.DataFrame(
    columns=[
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Vol",
    ]
)
google_df = pd.DataFrame(
    columns=[
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Vol",
    ]
)


# 发布者进程：获取股票价格并发布消息
def fetch_stock_price(queue_name, ticker_symbol):
    while True:
        stock = yf.Ticker(ticker_symbol)
        ochlv_df = stock.history(period="1d").iloc[-1]
        message = ochlv_df
        publish(queue_name, message)
        print(f"Published to {queue_name}: {message}")
        time.sleep(10)  # 每10秒获取一次最新股价


# 使用装饰器注册回调函数
@subscribe("apple")
def process_apple_message(message):
    apple_df = pd.concat([apple_df, message], ignore_index=True)
    print(f"Apple DataFrame Updated:\n{apple_df}")


@subscribe("google")
def process_google_message(message):
    google_df = pd.concat([google_df, message], ignore_index=True)
    print(f"Google DataFrame Updated:\n{google_df}")


if __name__ == "__main__":
    # 创建并启动发布者进程
    publisher_apple = multiprocessing.Process(
        target=fetch_stock_price, args=("apple", "AAPL")
    )
    publisher_google = multiprocessing.Process(
        target=fetch_stock_price, args=("google", "GOOGL")
    )

    publisher_apple.start()
    publisher_google.start()

    # 等待发布者进程结束
    publisher_apple.join()
    publisher_google.join()
