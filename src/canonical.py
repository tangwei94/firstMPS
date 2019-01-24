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

if __name__ == '__main__':
    L = 6
    H = ed.xy_hamilt(L)
    psi = ed.ground_state(H)[1]
    A_list, B_list, sigma = canonical(psi, L)

    # tests
    msg = 'the left normalization condition of A'
    ispassed = True
    print('--- test ' + msg + ' ---')
    for ix in range(L//2):
        As = A_list[ix]
        A0, A1 = np.asmatrix(As[0]), np.asmatrix(As[1])
        resultA = np.dot(A0.T.conj(), A0) + np.dot(A1.T.conj(), A1)
        if len(As) != 2 or not np.allclose(np.matlib.identity(len(resultA)), resultA): 
            print(msg + ': test failed')
            ispassed = False
    if ispassed:
        print(msg + ': test passed')
    else: 
        ispassed = True

    msg = 'the right normalization condition of B'
    print('--- test ' + msg + ' ---')
    for ix in range(L//2):
        Bs = B_list[ix]
        B0, B1 = np.asmatrix(Bs[0]), np.asmatrix(Bs[1])
        resultB = np.dot(B0, B0.T.conj()) + np.dot(B1, B1.T.conj())
        if len(Bs) != 2 or not np.allclose(np.matlib.identity(len(resultB)), resultB): 
            print(msg + ': test failed')
            ispassed = False
    if ispassed:
        print(msg + ': test passed')
    else: 
        ispassed = True  
        
    msg = 'the comparison with the ED solution'
    print('--- test ' + msg + ' ---')
    psi_mps = np.zeros(2**L, dtype=np.complex)
    for state in range(2**L):
        getbit = np.frompyfunc(lambda p: bitops.bget(state, L - 1 - p), 1, 1)
        spin_cfg = getbit(range(L))

        mat_prod = np.matlib.identity(1, dtype=np.complex)
        for site in range(L//2):
            spin = spin_cfg[site]
            mat_prod = mat_prod.dot(A_list[site][spin])
        mat_prod = mat_prod.dot(np.diag(sigma))
        for site in np.arange(L//2) + L//2:
            spin = spin_cfg[site]
            mat_prod = mat_prod.dot(B_list[L - site - 1][spin])
            
        psi_mps[state] = mat_prod[0,0]
    if not np.allclose(psi_mps, psi):
        ispassed = False
        print(msg + ': test failed')
        print(np.linalg.norm(psi_mps))
        for state in range(2**L):
            print(state, np.around([psi_mps[state], psi[state]], 3))
    if ispassed:
        print(msg + ': test passed')
    else: 
        ispassed = True
    