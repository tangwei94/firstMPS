import numpy as np 
import ed
import bitops

def xy_mpo():
    shape = (4, 4)
    M00, M01, M10, M11 = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    M00[0,0], M00[3,3] = 1, 1
    M01[0,1], M01[2,3] = 1, -0.5
    M10[0,2], M10[1,3] = 1, -0.5
    M11[0,0], M11[3,3] = 1, 1
    M = np.asarray([[M00, M01],[M10, M11]])
    return M
