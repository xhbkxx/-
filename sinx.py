import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei'是黑体的意思
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 生成数据
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# 多项式拟合的阶数
degree = 5

# 特征缩放（归一化）
x_powers = np.vstack([x ** i for i in range(degree + 1)]).T
x_max = np.max(x_powers, axis=0)
x_min = np.min(x_powers, axis=0)
epsilon = 1e-10  # 一个很小的正数，避免除0
x_normalized = (x_powers - x_min) / (x_max - x_min + epsilon)  # 归一化到[0, 1]范围（加epsilon防止除0）

# 初始化权重
w = np.random.randn(degree + 1)   # 初始权重
initial_learning_rate = 0.1  # 学习率
iterations = 1000000  # 迭代次数

# 动量初始化
momentum = np.zeros(degree + 1)
momentum_factor = 0.9

# 梯度下降
for iteration in range(iterations):
    # 计算多项式预测值
    y_pred = np.dot(x_normalized, w)

    # 计算损失和梯度
    error = y_pred - y
    loss = np.mean(error ** 2)

    # 防止损失值为nan
    if np.isnan(loss):
        print("Loss is NaN, stopping training.")
        break

        # 计算梯度
    gradient = np.dot(x_normalized.T, error) * 2 / len(x)

    # 更新动量
    momentum = momentum_factor * momentum + (1 - momentum_factor) * gradient

    # 更新权重
    w -= initial_learning_rate * momentum

    # 每5000次迭代打印损失和权重
    if iteration % 50000 == 0:
        print(f'Iteration {iteration}, Loss: {loss}')

    # 梯度下降拟合结果
y_fitted_gd = np.dot(x_normalized, w)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=' sin(x)曲线', color='blue', linewidth=2)
plt.plot(x, y_fitted_gd, label=' 拟合曲线', color='red', linestyle='--')
plt.title('曲线拟合')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.xlim(0, 2 * np.pi)
plt.show()