import numpy as np 
import ed
import bitops
import mpo

def test_xy_mpo():
    # test the mpo construction
    M = mpo.xy_mpo()
    L = 3
    H = ed.xy_hamilt(L).toarray()
    Hmps = np.zeros((2**L, 2**L), dtype=np.complex)
    for statei in range(2**L):
        getbiti = np.vectorize(lambda p: bitops.bget(statei, L - 1 - p))
        spin_cfgi = getbiti(range(L)) 
        for statej in range(2**L):
            getbitj = np.vectorize(lambda p: bitops.bget(statej, L - 1 - p))
            spin_cfgj = getbitj(range(L))

            Hij = M[spin_cfgi[0], spin_cfgj[0]]
            for ix in np.arange(L - 1) + 1:
                Hij = Hij.dot(M[spin_cfgi[ix], spin_cfgj[ix]])

            Hmps[statei, statej] = Hij[0, 3]

    assert np.allclose(Hmps, H)

        