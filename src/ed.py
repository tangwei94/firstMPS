''' ED for the XY chain '''

import numpy as np
from numpy import matlib
import scipy
from scipy import sparse
from scipy.sparse import linalg
import bitops

def xy_hamilt(n):
    # given the length of the xy chain, construct the hamiltonian matrix
    mat_dim = 2**n
    shape = (mat_dim, mat_dim)
    data, rows, cols = [], [], []
    for state_i in range(mat_dim):
        getbit = np.frompyfunc(lambda p: bitops.bget(state_i, n - 1 - p), 1, 1)
        spin_cfg = getbit(range(n))
        for site_i in range(n - 1):
            if spin_cfg[site_i] != spin_cfg[site_i + 1]:
                target = bitops.bflip(state_i, n - 1 - site_i)
                target = bitops.bflip(target, n - 1 - (site_i + 1))
                data.append(-0.5)
                rows.append(target)
                cols.append(state_i)
    H = scipy.sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.complex)
    return H

def ground_state(H): 
    # ground state solved by ED
    gs = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    E = gs[0][0]
    Psi = gs[1].transpose()[0]
    return E, Psi

if __name__=='__main__':
    H = xy_hamilt(6)
    gs = ground_state(H)
    print(np.around(gs[0], decimals=2))
    print(np.around(gs[1], decimals=2))
    