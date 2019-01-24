import numpy as np 
from numpy import linalg
import scipy
from scipy import sparse
from scipy.sparse import linalg
import sys
import ed, mpo
import contraction as contr
import spinadd as sadd
from canonical import canonical

class idmrg(object):
    def __init__(self, bondD, targetL):
        self.bondD = bondD
        self.targetL = targetL
        self.M = mpo.xy_mpo()

        if self.targetL % 2 != 0: 
            sys.exit('L must be even')

        initL = 2*int(np.log2(self.bondD)) + 2
        self.currL = initL
        
        H0 = ed.xy_hamilt(initL)
        psi0 = ed.ground_state(H0)[1]
        A_list, B_list, sigma = canonical(psi0, initL)

        # truncate
        A_list[initL//2 - 1] = np.compress([1]*self.bondD, A_list[initL//2 - 1], axis=2)
        B_list[initL//2 - 1] = np.compress([1]*self.bondD, B_list[initL//2 - 1], axis=1)
        sigma = np.compress([1]*self.bondD, sigma)

        self.As, self.Bs = A_list[initL//2 - 1], B_list[initL//2 - 1]
        self.As1, self.Bs1 = A_list[initL//2 - 2], B_list[initL//2 - 2]
        
        self.sigma = sigma

        self.Cl = np.reshape([1,0,0,0], (1,1,4,1)) 
        self.Cr = np.reshape([0,0,0,1], (1,4,1,1))
        for ix in range(initL//2):
            self.Cl = contr.contractL(self.Cl, A_list[ix], self.M)
            self.Cr = contr.contractR(self.Cr, B_list[ix], self.M)

    def growby4(self):
        newAs, newBs, initpsi = sadd.init_newpsi(self.sigma, self.As, self.As1, self.Bs, self.Bs1)
        newCl = contr.contractL(self.Cl, newAs, self.M)
        newCr = contr.contractR(self.Cr, newBs, self.M)
        
        newClH = np.einsum('adef,bceg->abdgcf', newCl, self.M)
        newCrH = np.einsum('abce,defz->adcbfz', self.M, newCr)

        shapeCl = newClH.shape
        newClH = np.reshape(newClH, (shapeCl[1]*shapeCl[2], shapeCl[3], shapeCl[4]*shapeCl[5]))
        shapeCr = newCrH.shape
        newCrH = np.reshape(newCrH, (shapeCr[0]*shapeCr[1], shapeCr[2], shapeCr[3]*shapeCr[4]))

        H2 = np.einsum('acd,bce->abde', newClH, newCrH)
        dimH2 = H2.shape[0]*H2.shape[1]
        H2 = np.reshape(H2, (dimH2, dimH2))
        H2s = scipy.sparse.csr_matrix(H2)
        
        if not np.allclose(H2, H2.T.conj()):
            sys.exit(str(self.currL)+': H2 not Hermitian')

        initpsi = np.reshape(initpsi, (dimH2, 1))
        newpsi = scipy.sparse.linalg.eigsh(H2s, k=1, which='SA', v0=initpsi)[1]
        #newpsi = scipy.sparse.linalg.eigsh(H2s, k=1, which='SA')[1]
        newpsi = np.reshape(newpsi, (newClH.shape[0], newClH.shape[0]))
        u, s, vdag = np.linalg.svd(newpsi)

        # update
        self.As = np.reshape(u, (2, u.shape[0]//2, u.shape[1]))
        self.As1 = newAs
        self.Bs = np.reshape(vdag, (vdag.shape[0], 2, vdag.shape[1]//2))
        self.Bs = np.swapaxes(self.Bs, 0, 1)
        self.Bs1 = newBs
        self.sigma = s

        self.Cl = contr.contractL(newCl, self.As, self.M)
        self.Cr = contr.contractR(newCr, self.Bs, self.M)

if __name__ == '__main__':
    bondD = 6
    targetL = 100
    hello = idmrg(bondD, targetL)

    for ix in range(200):
        print(ix)
        hello.growby4()
        print(ix, np.around(hello.sigma, 4))