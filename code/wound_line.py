import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成伤口曲线的参数方程
def wound_curve(t):
    x = np.cos(t)
    y = np.sin(t) + 0.5*np.sin(2*t)
    z = -0.1*np.sin(4*t)
    # x = t
    # y = t**2
    # z = t**3
    return x, y, z

# 在伤口曲线周围生成随机点
n_samples = 200 # 随机点的总数
n_points = 100 # 伤口曲线的采样点数
dist_thresh = 0.05 # 随机点到曲线的最大距离
t_values = np.linspace(0, 2, n_points)
points = []
while len(points) < n_samples:
    t = np.random.uniform(0, 2) # 随机选择一个参数值
    x0, y0, z0 = wound_curve(t) # 计算对应的伤口曲线上的点
    n = np.random.poisson(5) # 随机生成一定数量的随机点
    xs, ys, zs = x0 + np.random.normal(0, 0.2, n), y0 + np.random.normal(0, 0.2, n), z0 + np.random.normal(0, 0.05, n)
    dists = np.sqrt((xs-x0)**2 + (ys-y0)**2 + (zs-z0)**2) # 计算随机点到伤口曲线上点的距离
    mask = dists < dist_thresh # 筛选出合法的随机点
    points.extend(np.array([xs[mask], ys[mask], zs[mask]]).T.tolist())
points = np.array(points)[:n_samples]

# 将伤口曲线和随机点绘制到三维空间中
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y, z = wound_curve(t_values)
ax.plot(x, y, z, label='curve')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, c='r', depthshade=True, label='noise')
ax.legend()
plt.show()
