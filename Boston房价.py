import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib import font_manager

# 设置中文字体
zh_font_path = 'C:/Windows/Fonts/simhei.ttf'  # 请将此路径替换为你系统中的中文字体路径
zh_font = font_manager.FontProperties(fname=zh_font_path)

# 读取CSV文件并提取数据
filename = 'D:/浏览器下载/boston.csv'  # 请将此路径替换为你的实际CSV文件路径
data = []
with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)

    # 跳过表头
    next(reader)

    for row in reader:
        data.append([float(value) for value in row])

# 转换为numpy数组
data = np.array(data)

# 提取所有特征（除最后一列房价以外）
X = data[:, :-1]  # 多维特征
y = data[:, -1]   # 房价中位数

# 标准化特征（Z-score 标准化）
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# 初始化线性回归参数
m = np.zeros(X.shape[1])  # 多维特征的回归系数初始化为0
b = 0  # 截距初始化为0
learning_rate = 0.001  # 学习率
epochs = 10000  # 最大迭代次数
tolerance = 1e-6  # 收敛条件

# 存储每次迭代的误差
error_list = []

# 定义计算误差（均方误差）函数
def compute_error(X, y, m, b):
    n = len(y)
    predictions = np.dot(X, m) + b  # 多维线性回归的预测公式
    error = predictions - y
    total_error = np.dot(error, error)  # 误差平方和
    return total_error / n  # 均方误差

# 极大似然估计的梯度计算
def log_likelihood_gradient(X, y, m, b):
    n = len(y)
    predictions = np.dot(X, m) + b
    error = predictions - y
    m_gradient = (2 / n) * np.dot(X.T, error)  # m的梯度
    b_gradient = (2 / n) * np.sum(error)       # b的梯度
    return m_gradient, b_gradient

# 梯度下降法来最大化似然函数
previous_error = compute_error(X, y, m, b)  # 初始误差
for epoch in range(epochs):
    m_gradient, b_gradient = log_likelihood_gradient(X, y, m, b)

    # 更新参数
    m -= learning_rate * m_gradient
    b -= learning_rate * b_gradient

    # 计算新的误差
    current_error = compute_error(X, y, m, b)
    error_list.append(current_error)  # 记录每次迭代的误差

    # 输出每1000次迭代的误差
    if epoch % 1000 == 0:
        print(f"迭代次数: {epoch}, 当前误差: {current_error}")

    # 检查是否满足收敛条件
    if abs(previous_error - current_error) < tolerance:
        print(f"收敛于迭代次数: {epoch}, 当前误差: {current_error}")
        break

    previous_error = current_error  # 更新误差

# 输出最终模型参数
print(f"最终模型参数 m: {m}")
print(f"最终模型截距 b: {b}")

# 预测函数（基于多维特征）
def predict(X_value):
    X_value = (X_value - X_mean) / X_std  # 标准化输入的特征
    return np.dot(X_value, m) + b

# 预测所有数据的房价
y_pred = np.dot(X, m) + b

# 图1: 实际房价与预测房价的关系图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y, y_pred, color='blue', label='预测值 vs 实际值')
m_fit, b_fit = np.polyfit(y, y_pred, 1)  # 拟合直线
plt.plot(y, m_fit * y + b_fit, color='red', label='趋势线', linewidth=2)
plt.title('实际房价与预测房价的对比', fontproperties=zh_font)
plt.xlabel('实际房价 (千美元)', fontproperties=zh_font)
plt.ylabel('预测房价 (千美元)', fontproperties=zh_font)
plt.legend(prop=zh_font)
plt.grid(True)

# 图2: 迭代次数与误差的关系图
plt.subplot(1, 2, 2)
plt.plot(range(len(error_list)), error_list, color='green', label='误差下降曲线')
plt.title('迭代次数与误差', fontproperties=zh_font)
plt.xlabel('迭代次数', fontproperties=zh_font)
plt.ylabel('误差', fontproperties=zh_font)
plt.legend(prop=zh_font)
plt.grid(True)

plt.tight_layout()
plt.show()
