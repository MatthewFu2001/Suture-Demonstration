import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

demo_data = pd.read_csv('./demo_trajectory/train_trajectory_for_discrete_dmp_straight.csv',header=None)
demo_data = demo_data.T
features = np.diff(demo_data,axis=0)
kmeans = KMeans(n_clusters=3)
kmeans.fit(features)
labels = kmeans.labels_
# print(labels)
motion_primitives = []
for i in range(len(np.unique(labels))):
    motion_prim = demo_data[1:][labels == i]
    motion_primitives.append(motion_prim)


motion_primitives_np = np.asarray(motion_primitives)
motion_primitives_df = pd.DataFrame(data=motion_primitives_np)
motion_primitives_df.to_csv('motion_primitives.csv')