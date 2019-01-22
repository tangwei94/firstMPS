''' initialize the iDMRG program with a 4-spin problem '''

import numpy as np
from numpy import matlib
import scipy
from scipy import sparse
import bitops

def xy_hamilt(n):
    # given the length of the xy chain, construct the hamiltonian matrix
    mat_dim = 2**n
    shape = (mat_dim, mat_dim)
    data, rows, cols = [], [], []
    for state_i in range(mat_dim):
        getbit = np.frompyfunc(lambda p: bitops.bget(state_i, p), 1, 1)
        spin_cfg = getbit(range(n))
        for site_i in range(n - 1):
            if spin_cfg[site_i] != spin_cfg[site_i + 1]:
                target = bitops.bflip(state_i, site_i)
                target = bitops.bflip(target, site_i + 1)
                data.append(-0.5)
                rows.append(target)
                cols.append(state_i)
    H = scipy.sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.complex)
    print(np.around(H.todense(), decimals=2))
    return H

if __name__=='__main__':
    xy_hamilt(4)
    