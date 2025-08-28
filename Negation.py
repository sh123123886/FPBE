import numpy as np  # 使用 numpy 来简化矩阵运算（可选）

# 设置输出格式，防止出现科学计数法
np.set_printoptions(precision=5, suppress=True)

# 定义矩阵 A
A = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1/3, 0, 0],
    [0, 1/3, 0, 0, 0],
    [0, 1/3, 1/3, 0, 1],
    [0, 1/3, 1/3, 1, 0]
])

# 定义初始列向量 B
B = np.array([[0.0], [0.3], [0.4], [0.1], [0.2]])

# 定义迭代次数
iterations = 8

# 存储每次迭代的结果
results = [B.copy()]  # 初始结果包含 B 的副本

# 执行迭代
for _ in range(iterations):
    B = np.dot(A, B)  # 使用 numpy 的 dot 函数进行矩阵乘法
    results.append(B.copy())  # 存储结果

# 输出每次迭代的结果，保留6位小数
for i, result in enumerate(results):
    print(f"Iteration {i}:")
    print(result)  # 输出结果时不再使用 np.round，已在 set_printoptions 中设置
    print()
