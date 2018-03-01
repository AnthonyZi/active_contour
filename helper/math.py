import numpy as np

def cartesian2(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

def rotate_R2vector(vec,theta):
    theta = np.radians(theta)
    c,s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c,-s],[s,c]])
    v = vec.reshape(-1,1)
    return np.array(R*v).flatten()

