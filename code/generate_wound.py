import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import time

# 假设您已经定义了伤口曲线函数的参数和数据范围

# 定义生成伤口曲线的函数
def generate_wound_curve(num_points, noise_std, seed):
    # 接下来我们会生成一个随机的伤口曲线
    np.random.seed(i)
    # 随机生成伤口曲线的基本形状
    num_control_points = 10  # 设置控制点数目
    t = np.linspace(0, 1, num_control_points)  # 均匀分布t
    control_points = np.random.rand(num_control_points, 3)  # 生成随机的3D控制点

    # 利用三次B样条曲线函数生成伤口曲线
    degree = 3  # 使用三次B样条曲线
    knots = np.linspace(0, 1, num_control_points+degree-1)  # 均匀分布节点

    # 随机生成拟合误差
    error = np.random.normal(0, 0.05, num_control_points)

    # 设置样条函数
    spline = BSpline(knots, control_points, degree)
    # 均匀生成需要经过的点的序列
    t = np.linspace(0, 1, num_points)
    
    # 基于随机生成的控制点的样条函数生成伤口曲线
    curve_points = spline(t) + noise_std*np.random.randn(num_points, 3)

    return curve_points

# 生成10条随机伤口曲线并可视化
num_wounds = 10

fig = plt.figure(figsize=(8, 6))

for i in range(num_wounds):
    curve_points = generate_wound_curve(num_points=50, noise_std=0.05, seed=i)
    ax = fig.add_subplot(2, 5, i+1, projection='3d')
    ax.plot(curve_points[:,0], curve_points[:,1], curve_points[:,2], '-k')
    ax.set_title(f"Wound {i+1}")
    
plt.show()
