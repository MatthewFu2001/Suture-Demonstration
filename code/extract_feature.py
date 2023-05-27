# 利用高斯混合模型从包含三维坐标序列的示教数据中提取特征

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from sklearn.mixture import GaussianMixture

# 假设您已经生成了经过点的序列pts
# 接下来我们会根据这些经过点形成缝合曲线

# 首先，利用B样条曲线对经过点进行平滑拟合
num_control_points = len(pts)

degree = 3  # 使用三次B样条曲线

knots = np.linspace(0, 1, num_control_points+degree-1)  # 均匀分布节点

# 设置样条函数
spline = BSpline(knots, pts, degree)

# 定义计算曲线切向量的函数
def calc_tangent_vec(pts, t):
    # 对点序列进行差分（各个点之间的连线的斜率）
    tangent = np.diff(pts, axis=0)
    # 将斜率归一化为单位向量
    tangent /= np.linalg.norm(tangent, axis=1)[:,None]
    # 对第一个点和最后一个点的切向量进行特殊处理
    tangent = np.vstack([tangent[0], tangent, tangent[-1]])
    return spline(t), tangent

# 定义计算缝合线方向向量的函数
def calc_stitch_vec(tangent, t):
    # 利用外积计算与切向量垂直的缝合线向量
    stitch_normal = np.cross(tangent[:-1], tangent[1:])
    # 归一化后即得到缝合线方向向量
    stitch_vec = stitch_normal/np.linalg.norm(stitch_normal, axis=1)[:,None]
    stitch_vec = np.vstack([stitch_vec[0], stitch_vec, stitch_vec[-1]])
    # 根据t值插值获得每个点的缝合线方向向量
    return spline(t), stitch_vec

# 定义生成缝合轨迹的函数
def generate_stitch_curve(pts, num_points):
    # 中间参数t和切向量tangent
    t = np.linspace(0, 1, len(pts))
    pts_interp, tangent = calc_tangent_vec(pts, t)

    # 根据切向量计算得到缝合线方向向量
    pts_interp, stitch_vec = calc_stitch_vec(tangent, t)

    # 沿缝合线方向向量移动固定距离得到缝合点序列
    stitch_points = pts_interp + stitch_vec * 0.05
    
    # 使用B样条曲线对缝合点序列进行平滑，得到缝合轨迹
    stitch_spline = BSpline(t, stitch_points, degree)
    t_new = np.linspace(0, 1, num_points)
    return stitch_spline(t_new)

# 生成平滑的缝合轨迹并可视化
stitch_curve = generate_stitch_curve(pts, num_points=50)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot(pts[:,0], pts[:,1], pts[:,2], '-k')
ax.plot(stitch_curve[:,0], stitch_curve[:,1], stitch_curve[:,2], '-r')
# 接下来我们会利用高斯混合模型从示教数据中提取特征

# 首先，我们需要将示教数据转换为高斯混合模型需要的格式
# 由于我们的示教数据是三维坐标序列，因此我们需要将其转换为二维矩阵
# 二维矩阵的每一行代表一个三维坐标点，每一列代表一个坐标轴
# 例如，示教数据的第一个点的坐标为(0.1, 0.2, 0.3)，那么转换后的矩阵的第一行为[0.1, 0.2, 0.3]
# 由于示教数据中包含多条示教轨迹，因此我们需要将这些示教轨迹拼接成一个大的矩阵
# 例如，示教数据中包含两条示教轨迹，第一条示教轨迹包含三个点，第二条示教轨迹包含四个点
# 那么转换后的矩阵的第一行为第一条示教轨迹的第一个点的坐标，第二行为第一条示教轨迹的第二个点的坐标，第三行为第一条示教轨迹的第三个点的坐标，第四行为第二条示教轨迹的第一个点的坐标，第五行为第二条示教轨迹的第二个点的坐标，第六行为第二条示教轨迹的第三个点的坐标，第七行为第二条示教轨迹的第四个点的坐标
# 请注意，示教数据中的每个点都是三维坐标，因此转换后的矩阵的每一行都是三维向量

# 二维矩阵
demo_data = np.array([[0.1, 0.2, 0.3],
                        [0.2, 0.3, 0.4],
                        [0.3, 0.4, 0.5],
                        [0.4, 0.5, 0.6],
                        [0.5, 0.6, 0.7],
                        [0.6, 0.7, 0.8]])

# 对demo_data利用高斯混合模型提取特征
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(demo_data)

# 提取得到的特征
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

# 可视化提取得到的特征
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(means[:,0], means[:,1], means[:,2], c='r', marker='o', s=100)
ax.scatter(demo_data[:,0], demo_data[:,1], demo_data[:,2], c='k', marker='o', s=100)









