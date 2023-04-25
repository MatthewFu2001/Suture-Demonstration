import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

primitives_data = pd.read_csv('motion_primitives.csv')
primitives_data = primitives_data.values
test_data = pd.read_csv('./demo_trajectory/train_trajectory_for_discrete_dmp_lc_20.csv',header=None)
test_data = test_data.T
test_feature = np.diff(test_data,axis=0)
test_timestamp = test_data[:,0]

test_labels = []
for i in range(len(test_feature)):
    distances = cdist(primitives_data,[test_feature[i]],'euclidean')
    nearest_prim_idx = np.argmin(distances)
    test_labels.append(nearest_prim_idx)


interpolated_data = []
for i in range(len(np.unique(test_labels))):
    label_idxs = np.where(np.array(test_labels)==i)[0]
    primitive_data = primitives_data[i]
    if len(label_idxs) == 1:
        interpolated_data.append(primitive_data)
    else:
        primitive_timestamp = np.arange(len(primitive_data))
        primitive_interp_fun = np.poly1d(np.polyfit(primitive_timestamp,primitive_data,1))
        interpolated_timestamp = test_timestamp[label_idxs]
        interpolated_data.append(primitive_interp_fun(interpolated_timestamp))

segmented_data = np.vstack(interpolated_data)
segmented_data_df = pd.DataFrame(data=segmented_data)
segmented_data_df = segmented_data_df.T
segmented_data_df.to_csv('segmented_data.csv')