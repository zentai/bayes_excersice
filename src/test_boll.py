import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import talib

def calculate_actual_price(df, wavelet='db1', level=4, window_size=14):
    """
    使用滑動窗口方法對傳入的OHLCV數據進行小波轉換，並計算出平滑後的實際價格
    """
    prices = df['Close'].values
    actual_prices = np.zeros_like(prices)

    for i in range(window_size, len(prices)):
        window = prices[i - window_size:i]
        coeffs = pywt.wavedec(window, wavelet, level=level)
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
        denoised_window = pywt.waverec(coeffs, wavelet)
        actual_prices[i] = denoised_window[-1]  # 只取最新一個值

    # 將最初的window_size內的actual_prices設置為NaN以避免誤導
    actual_prices[:window_size] = np.nan

    return actual_prices

def plot_bollinger_bands(df, actual_prices, window=7, num_std_dev=3):
    """
    繪製收盤價格及其布林通道，並比較經過小波轉換後的實際價格及其布林通道
    """
    close = df['Close']

    # 計算原始布林通道
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=window, nbdevup=num_std_dev, nbdevdn=num_std_dev, matype=0)

    # 計算實際價格的布林通道
    upperband_actual, middleband_actual, lowerband_actual = talib.BBANDS(actual_prices, timeperiod=window, nbdevup=num_std_dev, nbdevdn=num_std_dev, matype=0)

    # 繪圖
    plt.figure(figsize=(14, 8))
    
    # 繪製原始數據的布林通道（淺色虛線）
    plt.plot(df.index, close, label='Close Price', color='red', linestyle='--')
    plt.plot(df.index, upperband, label='Bollinger Upper Band (Original)', color='lightcoral', linestyle='--')
    # plt.plot(df.index, middleband, label='Bollinger Middle Band (Original)', color='lightgreen', linestyle='--')
    plt.plot(df.index, lowerband, label='Bollinger Lower Band (Original)', color='lightcoral', linestyle='--')

    # 繪製經過小波轉換後的實際價格及其布林通道
    plt.plot(df.index, actual_prices, label='Actual Price (Wavelet)', color='orange')
    # plt.plot(df.index, upperband_actual, label='Bollinger Upper Band (Actual)', color='purple')
    # plt.plot(df.index, middleband_actual, label='Bollinger Middle Band (Actual)', color='brown')
    # plt.plot(df.index, lowerband_actual, label='Bollinger Lower Band (Actual)', color='purple')

    plt.legend()
    plt.title('Bollinger Bands Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def main():
    from config import config
    DATA_DIR, SRC_DIR, REPORTS_DIR = config.data_dir, config.src_dir, config.reports_dir
    
    ccy = 'turbousdt'
    df = pd.read_csv(f'{DATA_DIR}/{ccy}_cached.csv')
    print(df)
    actual_prices = calculate_actual_price(df)
    plot_bollinger_bands(df, actual_prices)

if __name__ == "__main__":
    main()
