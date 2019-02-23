import numpy as np
from numpy import linalg
import ed, bitops
from canonical import canonical

def test_canonical():
    L = 6
    H = ed.xy_hamilt(L)
    psi = ed.ground_state(H)[1]
    A_list, B_list, sigma = canonical(psi, L)

    # test the left normalization condition of A
    for ix in range(L//2):
        As = A_list[ix]
        A0, A1 = np.asarray(As[0]), np.asarray(As[1])
        resultA = np.dot(A0.T.conj(), A0) + np.dot(A1.T.conj(), A1)
        assert len(As) == 2
        assert np.allclose(np.identity(len(resultA)), resultA)

    # test the right normalization condition of B
    for ix in range(L//2):
        Bs = B_list[ix]
        B0, B1 = np.asarray(Bs[0]), np.asarray(Bs[1])
        resultB = np.dot(B0, B0.T.conj()) + np.dot(B1, B1.T.conj())
        assert len(Bs) == 2
        assert np.allclose(np.identity(len(resultB)), resultB)

    # compare with the ED solution
    psi_mps = np.zeros(2**L, dtype=np.complex)
    for state in range(2**L):
        getbit = np.frompyfunc(lambda p: bitops.bget(state, L - 1 - p), 1, 1)
        spin_cfg = getbit(range(L))

        mat_prod = np.identity(1, dtype=np.complex)
        for site in range(L//2):
            spin = spin_cfg[site]
            mat_prod = mat_prod.dot(A_list[site][spin])
        mat_prod = mat_prod.dot(np.diag(sigma))
        for site in np.arange(L//2) + L//2:
            spin = spin_cfg[site]
            mat_prod = mat_prod.dot(B_list[L - site - 1][spin])
            
        psi_mps[state] = mat_prod[0,0]
    
    assert np.allclose(psi_mps, psi)
