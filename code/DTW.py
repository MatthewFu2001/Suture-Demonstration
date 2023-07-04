from scipy.spatial.distance import cdist

def dtw(template, new_data):
    # 计算距离矩阵
    dist = cdist(template, new_data)
    # 初始化矩阵
    M = np.zeros((dist.shape[0] + 1, dist.shape[1] + 1))
    for i in range(1, dist.shape[0] + 1):
        for j in range(1, dist.shape[1] + 1):
            M[i, j] = dist[i - 1, j - 1]
    # 计算最短路径
    for i in range(1, dist.shape[0] + 1):
        for j in range(1, dist.shape[1] + 1):
            M[i, j] += min(M[i - 1, j], M[i, j - 1], M[i - 1, j - 1])
    # 回溯路径
    i, j = dist.shape[0], dist.shape[1]
    path = [(i, j)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            min_idx = np.argmin([M[i - 1, j], M[i, j - 1], M[i - 1, j - 1]])
            if min_idx == 0:
                i -= 1
            elif min_idx == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    path.append((1, 1))
    # 反转路径
    path.reverse()
    return M[dist.shape[0] + 1, dist.shape[1] + 1], path

# 定义模板曲线和待匹配曲线
template = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
new_data = np.array([[1, 0, 0], [2, 1, 1], [1, 2, 2], [3, 3, 3]])

# 计算匹配距离和路径信息
dist, path = dtw(template, new_data)

# 输出匹配距离和路径信息
print(f"Matching Distance: {dist}")
print(f"Matching Path: {path}")

