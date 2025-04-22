import pandas as pd
import numpy as np
from pykalman import KalmanFilter

# 讀入資料
data = pd.read_csv('data/BTC-USD.csv', index_col=0, parse_dates=True)

# 計算指數移動平均線
ema_period = 10
ema_alpha = 2 / (ema_period + 1)
data['EMA'] = data['Close'].ewm(alpha=ema_alpha).mean()

# # 計算權重移動平均線
wma_period = 10
weights = np.arange(1, wma_period + 1)
weights = pd.Series(weights).astype(float)
weights /= weights.sum()
data['WMA'] = data['Close'].rolling(window=wma_period).apply(lambda x: (weights * x).sum(), raw=True)
print(data)
# 融合指數移動平均線和權重移動平均線
obs_mat = np.vstack([data['EMA'], data['WMA']]).T[:, np.newaxis]
kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, initial_state_mean=[0, 0],
                  initial_state_covariance=np.eye(2),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=0.05,
                  transition_covariance=0.001)
state_means, _ = kf.filter(data['Close'].values[:, np.newaxis])
data['KF'] = state_means[:, 0]

# 繪製圖表
data[['Close', 'EMA', 'WMA', 'KF']].plot(figsize=(10, 6))
