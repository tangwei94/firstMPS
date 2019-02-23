import numpy as np
from numpy import linalg
import ed, bitops

def canonical(Psi, n):
    # wavefunction Psi of a n-site lattice, transform into mixed canonical form
    # n has to be even number
    n = int(n)
    psi = np.asarray(Psi) / np.linalg.norm(Psi)
    U_list, Vdag_list = [], []

    for ix in range(n//2 - 1):
        mat = np.reshape(psi, (2**(ix+1), 2**(n-ix-1)))
        q, r = np.linalg.qr(mat)
        U_list.append(q)
        psi = r

    for ix in range(n//2 -1):
        mat = np.reshape(psi, (2**(n-ix-1), 2**(ix+1)))
        mat_dag = mat.T.conj()
        q, r = np.linalg.qr(mat_dag)
        Vdag_list.append(q.T.conj())
        psi = r.T.conj()

    mat = np.reshape(psi, (2**(n//2), 2**(n//2)))
    u, s, vdag = np.linalg.svd(mat)
    U_list.append(u)
    Vdag_list.append(vdag)

    # interchange indices
    A_list, B_list = [], []
    for ix in range(n//2):
        U, Vdag = U_list[ix], Vdag_list[ix]
        A = np.reshape(U, (U.shape[0]//2, 2, U.shape[1]))
        A = np.swapaxes(A, 0, 1)
        B = np.reshape(Vdag, (Vdag.shape[0], 2, Vdag.shape[1]//2))
        B = np.swapaxes(B, 0, 1)
        A_list.append(A)
        B_list.append(B)

    return A_list, B_list, s
