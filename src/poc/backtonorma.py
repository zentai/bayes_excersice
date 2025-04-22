import numpy as np

# 獲取收盤價數據
close_price = np.array([1.0, 1.5, 2.0, 1.5, 1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0])

# 計算收盤價的移動平均線
window_size = 5
ma = np.convolve(close_price, np.ones(window_size)/window_size, mode='valid')

# 計算移動平均線和收盤價的差值
delta = close_price[window_size-1:] - ma

# 計算標準差
std = np.std(delta)

# 設置開倉門檻
open_threshold = 2 * std

# 設置平倉門檻
close_threshold = std

# 初始化持倉狀態和資本金
holding = False
capital = 1000

# 迭代計算每個交易日的收益
for i in range(window_size-1, len(close_price)):
    # 如果當前沒有持倉，檢查是否符合開倉條件
    if not holding and delta[i-window_size+1] < -open_threshold:
        holding = True
        position = capital / close_price[i]
        print("開倉 - {} 價格: {:.2f}".format(i, close_price[i]))
    # 如果當前持倉，檢查是否符合平倉條件
    elif holding and delta[i-window_size+1] > close_threshold:
        holding = False
        profit = position * (close_price[i] - close_price[i-window_size+1])
        capital += profit
        print("平倉 - {} 價格: {:.2f}, 盈虧: {:.2f}, 資本金: {:.2f}".format(i, close_price[i], profit, capital))
    else:
        print(f'{i}, not match')