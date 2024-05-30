import numpy as np

def construct_matrix(x, y):
    return np.hstack((x.reshape((x.size, 1)), y.reshape((y.size, 1)))) # C_contiguous