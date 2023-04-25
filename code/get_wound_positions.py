import time
import numpy as np

import sys
sys.path.append('./UR5/VREP_RemoteAPIs')
import sim

sim.simxFinish(-1)  # 确保没有隐式连接
clientID = sim.simxStart('127.0.0.1', 19999, True, False, 5000, 5)  # 模拟器连接
if clientID != -1:
    print("Connected to remote API server")

    # 获取Graph对象句柄
    error, graph_handle = sim.simxGetObjectHandle(clientID, 'Graph', sim.simx_opmode_blocking)


    # 获取Graph起始点和终止点坐标
    error, x_start, y_start, x_end, y_end = sim.simxGetGraphObjectPositions(clientID, graph_handle, sim.simx_opmode_blocking)
    if error == sim.simx_return_ok:
        print("Start point coordinates: ({}, {})".format(x_start, y_start))
        print("End point coordinates: ({}, {})".format(x_end, y_end))
    else:
        print("Error occurred while getting Graph object positions")
        print("Error code:", error)

    sim.simxFinish(clientID)
else:
    print("Failed to connect to remote API server")
