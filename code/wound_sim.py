import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import BSpline
import time
import pandas as pd
import sys
sys.path.append('./DMP')
from dmp_discrete import dmp_discrete
matplotlib.use('TkAgg')
import sympy as sp

# 生成伤口曲线的参数方程
def wound_curve(ti):
    # x = np.cos(ti) + 0.6*np.cos(2*ti)
    # y = np.sin(ti) + 0.5*np.sin(2*ti)
    # z = 0.0*ti
    # z = -0.1*np.sin(4*ti)
    # x = ti
    # y = ti**2
    # z = ti**3
    mean = np.random.uniform(-2,2)
    std = np.random.uniform(0.5,1.5)
    amp = np.random.uniform(0.5,1.5)
    coeffs = np.random.uniform(-1,1,size=4)
    x = amp * np.exp(-((ti-mean)**2/(2*std**2)))
    y = np.sin(ti) + 0.5*np.sin(2*ti)
    z = -0.1*np.sin(4*ti)
    return x, y, z, mean, std, amp

def curve_normal(ti,mean,std,amp):
    x,y,z,t = sp.symbols('x y z t')
    # x = sp.cos(t)+0.6*sp.cos(2*ti)
    # y = sp.sin(t) + 0.5*sp.sin(2*t)
    # z = -0.1*sp.sin(4*t)
    # z = 0.0
    # x = t
    # y = t**2
    # z = t**3
    x = amp * sp.exp(-((ti-mean)**2/(2*std**2)))
    y = sp.sin(ti) + 0.5*sp.sin(2*ti)
    z = -0.1*sp.sin(4*ti)
    x_t = sp.diff(x,t)
    y_t = sp.diff(y,t)
    z_t = sp.diff(z,t)
    # z_t = 0.0
    dx_dt = float(x_t.evalf(subs={t:ti}))
    dy_dt = float(y_t.evalf(subs={t:ti}))
    dz_dt = float(z_t.evalf(subs={t:ti}))
    # dz_dt = 0.0
    d1 = np.array([dx_dt,dy_dt,dz_dt])
    dx_dt = float(x_t.evalf(subs={t:ti+0.02}))
    dy_dt = float(y_t.evalf(subs={t:ti+0.02}))
    dz_dt = float(z_t.evalf(subs={t:ti+0.02}))
    # normal = np.array([dy_dt,-dx_dt,0])
    d2 = np.array([dx_dt,dy_dt,dz_dt])
    delta = np.cross(d2,d1)
    normal = np.cross(delta,d1)
    normal_norm = np.linalg.norm(normal)
    normal_unit = normal/normal_norm
    return normal_unit


n_points = 100
num_points = 20
# 生成需要经过的等间距点的序列
t = np.linspace(0, 2, num_points)
t_values = np.linspace(0, 2, n_points)
x, y, z = wound_curve(t_values)
xx, yy, zz = wound_curve(t)
# 计算每个点的左右偏移向量，并将其与基点坐标相加以得到偏移点的坐标
displacement_distance = 0.2 # 设置偏移距离

left_points = []
right_points = []

normal_vec = []
for ti in t:
    normal_ = curve_normal(ti)
    normal_vec.append(normal_)

for i in range(num_points):
    normal = normal_vec[i] # 获取第 i 个点的法向量
    # up = np.array([0, 0, 1])  # 选定一个向上的向量作为参考
    # perpendicular = np.cross(up, normal)  # 计算垂直于参考向量的向量
    # perpendicular /= np.linalg.norm(perpendicular)  # 归一化垂向量
    left_point = np.array([xx[i], yy[i], zz[i]]) + displacement_distance * normal  # 计算左偏移点
    right_point = np.array([xx[i], yy[i], zz[i]]) - displacement_distance * normal  # 计算右偏移点
    left_points.append(left_point)
    right_points.append(right_point)


df_left = pd.read_csv('./demo_trajectory/straight_line_left_right.csv', header=None)
reference_trajectory_left = np.array(df_left)
data_dim = reference_trajectory_left.shape[0]
data_len = reference_trajectory_left.shape[1]
reproduced_trajectory_record_left_x = np.zeros((data_len, num_points-1))
reproduced_trajectory_record_left_y = np.zeros((data_len, num_points-1))
reproduced_trajectory_record_left_z = np.zeros((data_len, num_points-1))
df_right = pd.read_csv('./demo_trajectory/straight_line_right_left.csv', header=None)
reference_trajectory_right = np.array(df_right)
data_dim_right = reference_trajectory_right.shape[0]
data_len_right = reference_trajectory_right.shape[1]
reproduced_trajectory_record_right_x = np.zeros((data_len_right, num_points-1))
reproduced_trajectory_record_right_y = np.zeros((data_len_right, num_points-1))
reproduced_trajectory_record_right_z = np.zeros((data_len_right, num_points-1))
dmp = dmp_discrete(n_dmps=data_dim, n_bfs=1000, dt=1.0/data_len)
dmp.learning(reference_trajectory_left)
right_dmp = dmp_discrete(n_dmps=data_dim_right, n_bfs=1000, dt=1.0/data_len_right)
right_dmp.learning(reference_trajectory_right)
for i in range(num_points-1):
    if i%2 == 0:
        initial_pos = left_points[i].tolist()
        goal_pos = right_points[i+1].tolist()
        reproduced_trajectory, _, _ = dmp.reproduce(initial=initial_pos, goal=goal_pos)
        data_len = reproduced_trajectory.shape[0]
        reproduced_trajectory_record_left_x[:,i] = reproduced_trajectory[:,0]
        reproduced_trajectory_record_left_y[:,i] = reproduced_trajectory[:,1]
        reproduced_trajectory_record_left_z[:,i] = reproduced_trajectory[:,2]
    else:
        initial_pos = right_points[i].tolist()
        goal_pos = left_points[i+1].tolist()
        reproduced_trajectory, _, _ = right_dmp.reproduce(initial=initial_pos, goal=goal_pos)
        data_len = reproduced_trajectory.shape[0]
        reproduced_trajectory_record_right_x[:,i] = reproduced_trajectory[:,0]
        reproduced_trajectory_record_right_y[:,i] = reproduced_trajectory[:,1]
        reproduced_trajectory_record_right_z[:,i] = reproduced_trajectory[:,2]
    
    

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.legend()
ax.set_xlabel('x/dm')
ax.set_ylabel('y/dm')
ax.set_zlabel('z/dm')
x, y, z = wound_curve(t_values)
ax.plot(x, y, z, label='curve')
for i in range(num_points-1):
    if i%2 == 0:
        ax.plot(reproduced_trajectory_record_left_x[:,i], reproduced_trajectory_record_left_y[:,i], reproduced_trajectory_record_left_z[:,i], '--')
    else:
        ax.plot(reproduced_trajectory_record_right_x[:,i], reproduced_trajectory_record_right_y[:,i], reproduced_trajectory_record_right_z[:,i], '--')
ax.legend()
plt.show()