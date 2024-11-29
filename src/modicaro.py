import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取对账单数据
df = pd.read_csv("reports/adausdt1min_atr15bw15up15lw15_cut0.9pnl3ext2stp7.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# 确保'sProfit'列存在并转换为数值
df["sProfit"] = pd.to_numeric(df["sProfit"], errors="coerce")
df.dropna(subset=["sProfit"], inplace=True)

# 设置蒙特卡罗模拟参数
num_simulations = 1000  # 模拟路径数
num_days = 252 * 5  # 模拟天数（或者以交易次数来替代）

# 初始化资金曲线模拟结果
initial_capital = 10000  # 初始资金
simulations = np.zeros((num_days, num_simulations))

# 计算每次交易的收益率（假设df['Capital']是每次交易前的总资金）
df["Return_pct"] = df["sProfit"] / df["sCash"]

# 进行蒙特卡罗模拟（基于收益率）
for i in range(num_simulations):
    simulated_returns = np.random.choice(df["Return_pct"], num_days, replace=True)
    simulated_curve = [initial_capital]
    for r in simulated_returns:
        simulated_curve.append(simulated_curve[-1] * (1 + r))
    simulations[:, i] = simulated_curve[1:]

# 绘制资金曲线模拟结果
plt.figure(figsize=(14, 7))
plt.plot(simulations, color="lightgray", linewidth=0.5)
plt.title("Monte Carlo Simulation of Future Capital Curves Based on sProfit")
plt.xlabel("Days")
plt.ylabel("Capital")
plt.show()
