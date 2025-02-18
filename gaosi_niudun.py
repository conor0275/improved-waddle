import numpy as np

# 生成模拟数据
np.random.seed(0)  # 设置随机种子以确保结果可复现
x_data = np.linspace(0, 1, 100)  # 在 [0, 1] 区间内生成 100 个等间距点作为自变量 x
y_true = 2 * np.exp(-1 * x_data) + 0.5  # 真实函数 y = 2 * exp(-x) + 0.5
y_data = y_true + np.random.normal(scale=0.1, size=len(x_data))  # 添加噪声，模拟实际观测值

# 初始参数猜测
theta = np.array([1.0, -0.5, 0.0])  # 初始参数 a, b, c 的猜测值
max_iter = 100  # 最大迭代次数
tolerance = 1e-6  # 收敛条件：参数更新量小于该值时认为已收敛

for iter in range(max_iter):
    a, b, c = theta  # 解包当前参数值
    
    # 计算残差（预测值与实际值之差）
    residual = y_data - (a * np.exp(b * x_data) + c)
    
    # 计算雅可比矩阵 J
    J = np.zeros((len(x_data), 3))
    J[:, 0] = -np.exp(b * x_data)       # dr/da: 对 a 的偏导数
    J[:, 1] = -a * x_data * np.exp(b * x_data)  # dr/db: 对 b 的偏导数
    J[:, 2] = -1                        # dr/dc: 对 c 的偏导数
    
    # 求解线性最小二乘问题 JΔθ ≈ -residual
    delta, _, _, _ = np.linalg.lstsq(J, -residual, rcond=None)  # 计算参数增量 Δθ
    
    # 更新参数
    theta += delta
    
    # 检查是否满足收敛条件
    if np.linalg.norm(delta) < tolerance:
        print(f"Converged at iteration {iter}")  # 如果满足收敛条件，输出迭代次数并退出循环
        break

print("Estimated parameters:", theta)  # 输出最终估计的参数值