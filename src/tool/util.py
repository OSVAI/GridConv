import numpy as np
import math

def dict2list(d):
    # convert dict with list items to list with dict items.
    # eg. DL = {'a': [0, 1], 'b': [2, 3]}, LD=[{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
    # DL to LD
    l = [dict(zip(d,t)) for t in zip(*d.values())]
    return l

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def rearrange_by_key(array, guide_info, guide_key='action'):
    arr_rearrange = {}
    for i in range(len(array)):
        key = guide_info[i][guide_key]
        if key not in arr_rearrange:
            arr_rearrange[key] = []
        arr_rearrange[key].append(array[i])
    for key in arr_rearrange:
        arr_rearrange[key] = np.array(arr_rearrange[key])
    return arr_rearrange

def semantic_grid_trans(src_graph_pose):
    assert len(src_graph_pose.shape) == 3 # B*J*C
    batch_size, _, C = src_graph_pose.shape
    grid_pose = np.zeros([batch_size, 5, 5, C])
    grid_pose[:, 0] = src_graph_pose[:, [7, 7, 7, 7, 7]]
    grid_pose[:, 1] = src_graph_pose[:, [0, 8, 8, 8, 0]]
    grid_pose[:, 2] = src_graph_pose[:, [1, 14, 0, 11, 4]]
    grid_pose[:, 2, 2] = src_graph_pose[:, [8, 9]].mean(1)  # midpoint of neck and nose

    grid_pose[:, 3] = src_graph_pose[:, [2, 15, 9, 12, 5]]
    grid_pose[:, 4] = src_graph_pose[:, [3, 16, 10, 13, 6]]

    grid_pose = grid_pose.transpose([0, 3, 1, 2])   # B*C*5*5

    return grid_pose

def inverse_semantic_grid_trans(src_grid_pose):
    batch_size, C = src_grid_pose.shape[:2]
    src_grid_pose = src_grid_pose.transpose([0, 2, 3, 1])  # B*5*5*C

    graph_pose = np.zeros([batch_size, 17, C])
    graph_pose[:, 7] = src_grid_pose[:, 0].mean(axis=1)
    graph_pose[:, 0] = src_grid_pose[:, 1, [0, 4]].mean(axis=1)
    graph_pose[:, 8] = src_grid_pose[:, 1, [1, 2, 3]].mean(axis=1)
    graph_pose[:, [1, 14, 11, 4]] = src_grid_pose[:, 2, [0, 1, 3, 4]]
    graph_pose[:, [2, 15, 9, 12, 5]] = src_grid_pose[:, 3]
    graph_pose[:, [3, 16, 10, 13, 6]] = src_grid_pose[:, 4]


    return graph_pose

def get_temperature(start_epoch, cur_epoch, total_epoch, cur_iter, total_iter, method, max_temp, pow_x=10,
                    increase=False):

    if cur_epoch >= total_epoch:
        return 1

    ratio = ((cur_epoch - start_epoch) + (cur_iter / total_iter)) / (total_epoch - start_epoch)

    if not increase:
        ratio = 1.0 - ratio

    if method == 'linear':
        return 1 + ratio * (max_temp-1)
    elif method == 'exp':
        return math.exp(ratio * max_temp)
    elif method == 'pow':
        return math.pow(pow_x, ratio * max_temp)
    else:
        raise ValueError("Invalid choice for temperature")

def get_procrustes_transformation(X, Y, compute_optimal_scale=False):
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c