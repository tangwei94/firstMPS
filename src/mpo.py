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

if __name__ == '__main__':
    M = xy_mpo()
    # check the mpo construction
    msg = 'the mpo construction'
    print('--- test ' + msg + ' ---')

    L = 3
    H = ed.xy_hamilt(L).todense()
    Hmps = np.zeros((2**L, 2**L), dtype=np.complex)
    for statei in range(2**L):
        getbiti = np.frompyfunc(lambda p: bitops.bget(statei, L - 1 - p), 1, 1)
        spin_cfgi = getbiti(range(L)) 
        for statej in range(2**L):
            getbitj = np.frompyfunc(lambda p: bitops.bget(statej, L - 1 - p), 1, 1)
            spin_cfgj = getbitj(range(L))

            Hij = M[spin_cfgi[0], spin_cfgj[0]]
            for ix in np.arange(L - 1) + 1:
                Hij = Hij.dot(M[spin_cfgi[ix], spin_cfgj[ix]])

            Hmps[statei, statej] = Hij[0, 3]

    if np.allclose(Hmps, H):
        print(msg + ' test passed')
    else:
        print(msg + ' test failed')  
        print(Hmps)
        print(' ')
        print(H)
        