import numpy as np 
from numpy import linalg
import ed
from canonical import canonical
import contraction as contr
import warnings
import spinadd as sadd

def test_spinadd():
    L = 10
    H = ed.xy_hamilt(L)
    E, psi = ed.ground_state(H)
    A_list, B_list, sigma = canonical(psi, L)

    newAs, newBs, newpsi = sadd.init_newpsi(sigma, A_list[L//2-1], B_list[L//2-1])
    A_list += [newAs]
    B_list += [newBs]

    assert contr.check_Lnorm(A_list)
    assert contr.check_Rnorm(B_list)
