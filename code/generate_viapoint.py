import numpy as np
from scipy.spatial import distance

# 假设您已经定义了伤口曲线样条函数和需要经过的点数num_points以及每个点的法向量normal_vec
# 接下来我们会在伤口曲线上均匀生成num_points个点，并对每个点计算左右偏移点

# 生成需要经过的等间距点的序列
t = np.linspace(0, 1, num_points)

# 从伤口曲线样条函数中获取曲线上的点的坐标
x = spline(t)
y = spline(t)
z = spline(t)

# 计算每个点的左右偏移向量，并将其与基点坐标相加以得到偏移点的坐标
displacement_distance = 0.2 # 设置偏移距离 
for i in range(num_points):
    normal = normal_vec[i] # 获取第 i 个点的法向量
    up = np.array([0, 0, 1])  # 选定一个向上的向量作为参考
    perpendicular = np.cross(up, normal)  # 计算垂直于参考向量的向量
    perpendicular /= np.linalg.norm(perpendicular)  # 归一化垂向量
    

    left_point = np.array([x[i], y[i], z[i]]) + displacement_distance * perpendicular  # 计算左偏移点
    right_point = np.array([x[i], y[i], z[i]]) - displacement_distance * perpendicular  # 计算右偏移点
    
    print("Point ", i, ": (", x[i], ",", y[i], ",", z[i], ")  left: (", left_point[0], ",", left_point[1], ",", left_point[2], ") right: (", right_point[0], ",", right_point[1], ",", right_point[2], ")")
