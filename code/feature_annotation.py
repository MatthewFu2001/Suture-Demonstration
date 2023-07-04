import time
import numpy as np
import pandas as pd

import sys
sys.path.append('./UR5/VREP_RemoteAPIs')
import sim as vrep_sim

sys.path.append('./UR5')
from UR5SimModel import UR5SimModel

sys.path.append('./DMP')
from dmp_discrete import dmp_discrete
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from collections import OrderedDict


#%% program
print ('Program started')

# ------------------------------- Connect to VREP (CoppeliaSim) ------------------------------- 
vrep_sim.simxFinish(-1) # just in case, close all opened connections
while True:
    client_ID = vrep_sim.simxStart('127.0.0.1', 19999, True, False, 5000, 5) # Connect to CoppeliaSim
    if client_ID > -1: # connected
        print('Connect to remote API server.')
        break
    else:
        print('Failed connecting to remote API server! Try it again ...')

# Pause the simulation
# res = vrep_sim.simxPauseSimulation(client_ID, vrep_sim.simx_opmode_blocking)

delta_t = 0.01 # simulation time step
# Set the simulation step size for VREP
vrep_sim.simxSetFloatingParameter(client_ID, vrep_sim.sim_floatparam_simulation_time_step, delta_t, vrep_sim.simx_opmode_oneshot)
# Open synchronous mode
vrep_sim.simxSynchronous(client_ID, True) 
# Start simulation
vrep_sim.simxStartSimulation(client_ID, vrep_sim.simx_opmode_oneshot)

# ------------------------------- Initialize simulation model ------------------------------- 
UR5_sim_model = UR5SimModel()
UR5_sim_model.initializeSimModel(client_ID)


return_code, initial_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'initial', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get initial dummy handle ok.')

return_code, goal_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'goal', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get goal dummy handle ok.')

return_code, UR5_target_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'UR5_target', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get UR5 target dummy handle ok.')

return_code, via1_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via1', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via1 dummy handle ok.')

return_code, via2_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via2', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via2 dummy handle ok.')

return_code, via3_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via3', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via3 dummy handle ok.')

return_code, via4_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via4', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via4 dummy handle ok.')

return_code, via5_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via5', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via5 dummy handle ok.')

return_code, via6_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via6', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via6 dummy handle ok.')

return_code, via7_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via7', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via7 dummy handle ok.')

# return_code, via8_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via8', vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via8 dummy handle ok.')

# return_code, via9_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via9', vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via9 dummy handle ok.')

# return_code, via10_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via10', vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via10 dummy handle ok.')

# return_code, via11_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via11', vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via11 dummy handle ok.')

# return_code, via12_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via12', vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via12 dummy handle ok.')

# return_code, via13_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via13', vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via13 dummy handle ok.')

time.sleep(0.1)

df_approach = pd.read_csv('./demo_trajectory/skill_approach_insert.csv', header=None)
reference_trajectory_approach = np.array(df_approach)
data_dim_1 = reference_trajectory_approach.shape[0]
data_len_1 = reference_trajectory_approach.shape[1]

dmp_approach = dmp_discrete(n_dmps=data_dim_1, n_bfs=1000, dt=1.0/data_len_1)
dmp_approach.learning(reference_trajectory_approach)

df_pull = pd.read_csv('./demo_trajectory/skill_pull.csv', header=None)
reference_trajectory_pull = np.array(df_pull)
data_dim_2 = reference_trajectory_pull.shape[0]
data_len_2 = reference_trajectory_pull.shape[1]

dmp_pull = dmp_discrete(n_dmps=data_dim_2, n_bfs=1000, dt=1.0/data_len_2)
dmp_pull.learning(reference_trajectory_pull)

df_sew = pd.read_csv('./demo_trajectory/skill_sew_1.csv', header=None)
reference_trajectory_sew = np.array(df_sew)
data_dim_3 = reference_trajectory_sew.shape[0]
data_len_3 = reference_trajectory_sew.shape[1]

dmp_sew = dmp_discrete(n_dmps=data_dim_3, n_bfs=1000, dt=1.0/data_len_3)
dmp_sew.learning(reference_trajectory_sew)

df_seg = pd.read_csv('./demo_trajectory/skill_sew_2.csv', header=None)
reference_trajectory_seg = np.array(df_seg)
data_dim_4 = reference_trajectory_seg.shape[0]
data_len_4 = reference_trajectory_seg.shape[1]

dmp_seg = dmp_discrete(n_dmps=data_dim_4, n_bfs=1000, dt=1.0/data_len_4)
dmp_seg.learning(reference_trajectory_seg)

return_code, initial_pos = vrep_sim.simxGetObjectPosition(client_ID,initial_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get initial pos ok.')

return_code, goal_pos = vrep_sim.simxGetObjectPosition(client_ID,goal_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get goal pos ok.')

return_code, via1_pos = vrep_sim.simxGetObjectPosition(client_ID,via1_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via1 pos ok.')

return_code, via2_pos = vrep_sim.simxGetObjectPosition(client_ID,via2_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via2 pos ok.')

return_code, via3_pos = vrep_sim.simxGetObjectPosition(client_ID,via3_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via3 pos ok.')

return_code, via4_pos = vrep_sim.simxGetObjectPosition(client_ID,via4_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via4 pos ok.')

return_code, via5_pos = vrep_sim.simxGetObjectPosition(client_ID,via5_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via5 pos ok.')

return_code, via6_pos = vrep_sim.simxGetObjectPosition(client_ID,via6_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via6 pos ok.')

return_code, via7_pos = vrep_sim.simxGetObjectPosition(client_ID,via7_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via7 pos ok.')

# return_code, via8_pos = vrep_sim.simxGetObjectPosition(client_ID,via8_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via8 pos ok.')

# return_code, via9_pos = vrep_sim.simxGetObjectPosition(client_ID,via9_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via9 pos ok.')

# return_code, via10_pos = vrep_sim.simxGetObjectPosition(client_ID,via10_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via10 pos ok.')

# return_code, via11_pos = vrep_sim.simxGetObjectPosition(client_ID,via11_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via11 pos ok.')

# return_code, via12_pos = vrep_sim.simxGetObjectPosition(client_ID,via12_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via12 pos ok.')

# return_code, via13_pos = vrep_sim.simxGetObjectPosition(client_ID,via13_dummy_handle,-1,vrep_sim.simx_opmode_blocking)
# if (return_code == vrep_sim.simx_return_ok):
#     print('get via13 pos ok.')



vrep_sim.simxStopSimulation(client_ID, vrep_sim.simx_opmode_blocking) # stop the simulation
vrep_sim.simxFinish(-1)  # Close the connection
print('Program terminated')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.legend()
ax.set_xlabel('x/dm')
ax.set_ylabel('y/dm')
ax.set_zlabel('z/dm')

reproduced_trajectory, _, _ = dmp_sew.reproduce(initial=initial_pos,goal=via1_pos)
ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],label='sew',color='orangered')

reproduced_trajectory, _, _ = dmp_seg.reproduce(initial=via1_pos,goal=via2_pos)
ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='orangered')

reproduced_trajectory, _, _ = dmp_pull.reproduce(initial=via2_pos,goal=via3_pos)
ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],label='pull',color='deepskyblue')

reproduced_trajectory, _, _ = dmp_approach.reproduce(initial=via3_pos,goal=via4_pos)
ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],label='insert',color='mediumorchid')

reproduced_trajectory, _, _ = dmp_sew.reproduce(initial=via4_pos,goal=via5_pos)
ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='orangered')

reproduced_trajectory, _, _ = dmp_seg.reproduce(initial=via5_pos,goal=via6_pos)
ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='orangered')

reproduced_trajectory, _, _ = dmp_pull.reproduce(initial=via6_pos,goal=via7_pos)
ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='deepskyblue')

reproduced_trajectory, _, _ = dmp_approach.reproduce(initial=via7_pos,goal=goal_pos)
ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='mediumorchid')

# reproduced_trajectory, _, _ = dmp_sew.reproduce(initial=via8_pos,goal=via9_pos)
# ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='orangered')

# reproduced_trajectory, _, _ = dmp_seg.reproduce(initial=via9_pos,goal=via10_pos)
# ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='orangered')

# reproduced_trajectory, _, _ = dmp_pull.reproduce(initial=via10_pos,goal=via11_pos)
# ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='deepskyblue')

# reproduced_trajectory, _, _ = dmp_approach.reproduce(initial=via11_pos,goal=via12_pos)
# ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='mediumorchid')

# reproduced_trajectory, _, _ = dmp_sew.reproduce(initial=via12_pos,goal=via13_pos)
# ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='orangered')

# reproduced_trajectory, _, _ = dmp_seg.reproduce(initial=via13_pos,goal=goal_pos)
# ax.plot(reproduced_trajectory[:,0],reproduced_trajectory[:,1],reproduced_trajectory[:,2],color='orangered')

ax.legend()
plt.show()