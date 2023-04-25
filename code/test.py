import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

primitives_data = pd.read_csv('motion_primitives.csv')
primitives_data = primitives_data.values

print(primitives_data[1][1])