import numpy as np 
import ed, mpo
from canonical import canonical
import contraction as contr 

def test_contraction():
    L = 6
    H = ed.xy_hamilt(L)
    E, Psi = ed.ground_state(H)
    A_list, B_list, sigma = canonical(Psi, L)

    assert contr.check_Lnorm(A_list)
    assert contr.check_Rnorm(B_list)

    M = mpo.xy_mpo()
    Cl = np.reshape([1,0,0,0], (1,1,4,1))
    for ix in range(L//2):
        Cl = contr.contractL(Cl, A_list[ix], M)
    Cr = np.reshape([0,0,0,1], (1,4,1,1))
    for ix in range(L//2):
        Cr = contr.contractR(Cr, B_list[ix], M)
    
    Cl = np.einsum('abcd,be->aecd', Cl, np.diag(sigma))
    Cl = np.einsum('abcd,de->abce', Cl, np.diag(sigma))
    Cl = Cl.flatten()
    Cr = Cr.flatten()
    Emps = Cl.dot(Cr)

    assert np.isclose(E, Emps)

