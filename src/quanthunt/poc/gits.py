import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def profits_to_pdf_with_max_density(profits):
    """
    Estimate a PDF using Kernel Density Estimation from a series of profit values and find the value with the highest density.

    :param profits: pandas Series containing profit values.
    :return: A tuple containing the x values (range of values), the estimated PDF values, and the value with the highest density.
    """
    # 将 Series 转换为 numpy 数组
    values = profits.to_numpy()

    # 使用核密度估计来近似PDF
    kde = gaussian_kde(values)

    # 生成用于可视化的值范围
    x = np.linspace(min(values), max(values), 1000)
    pdf = kde.evaluate(x)

    # 找到最大密度的值
    max_density_value = x[np.argmax(pdf)]

    return x, pdf, max_density_value


# 将示例数据更正为利润列表
profits_list = [
    0.0060767382366162526,
    0.00998166122551769,
    0.010063276483736727,
    0.017039923029852577,
    0.009885267209800208,
    0.010219384502096807,
    0.014451895163424044,
    0.013214083993611014,
    0.01612827698106889,
    0.013534307349505337,
    0.011858562633041636,
    0.011830061801303193,
    0.009215550873216793,
    0.009161706996763996,
    0.011680159490478736,
    0.013396632149593968,
    0.015180412107947738,
    0.014744718048660177,
    0.012020257403109191,
    0.01293990499504627,
    0.014213619726774152,
    0.012085908249565014,
    0.01042354346586194,
    0.008522690409879408,
    0.006622101741404096,
    0.006514498455667317,
    0.0069175365014997325,
    0.008335471964433294,
    0.0064687621767531844,
    0.0036643005745440327,
    -2.3392847081638024e-07,
]

# 将利润列表转换为 Pandas Series
profits_series = pd.Series(profits_list)

# 使用函数转换利润值为 PDF
x_values, pdf_values, max_density_profit = profits_to_pdf_with_max_density(
    profits_series
)

# 绘制PDF图
plt.plot(x_values, pdf_values)
plt.title("Estimated Probability Density Function (PDF) of Profits")
plt.xlabel("Profit")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# 输出最大密度的利润值
max_density_profit
