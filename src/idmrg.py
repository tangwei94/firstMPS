import numpy as np 
from numpy import linalg
import scipy
from scipy import sparse
from scipy.sparse import linalg
import warnings
import ed, mpo
import contraction as contr
import spinadd as sadd
from canonical import canonical
import h5py
import time

class idmrg(object):
    def __init__(self, bondD, targetL):
        self.bondD = bondD
        self.M = mpo.xy_mpo()
        self.targetL = targetL
        if self.targetL % 2 != 0: 
            warnings.warn('idmrg: L must be even. L modified to L+1')
            self.targetL += 1

        self.dataname = 'data/xy_dmrg_D'+str(self.bondD)+'.h5'

        initL = 2*int(np.log2(self.bondD)) + 2
        self.currL = initL
        
        H0 = ed.xy_hamilt(initL)
        psi0 = ed.ground_state(H0)[1]
        A_list, B_list, sigma = canonical(psi0, initL)
        
        # truncate
        A_list[initL//2 - 1] = np.compress([1]*self.bondD, A_list[initL//2 - 1], axis=2)
        B_list[initL//2 - 1] = np.compress([1]*self.bondD, B_list[initL//2 - 1], axis=1)
        sigma = np.compress([1]*self.bondD, sigma)
        if not contr.check_Lnorm(A_list) or not contr.check_Rnorm(B_list):
            warnings.warn('idrmg: normalization condition violated due to truncation')

        self.As, self.Bs = A_list[initL//2 - 1], B_list[initL//2 - 1]
        self.sigma = sigma

        hf = h5py.File(self.dataname, 'w')
        hf['parameters/bondD']=self.bondD
        hf['parameters/currL']=self.currL

        self.Cl = np.reshape([1,0,0,0], (1,1,4,1)) 
        self.Cr = np.reshape([0,0,0,1], (1,4,1,1))
        for ix in range(initL//2):
            self.Cl = contr.contractL(self.Cl, A_list[ix], self.M)
            self.Cr = contr.contractR(self.Cr, B_list[ix], self.M)

            # save As, Bs, Cl, Cr
            hf['data/As_'+str(ix)] = A_list[ix].flatten()
            hf['data/As_'+str(ix)].attrs['shape'] = A_list[ix].shape
            hf['data/Cl_'+str(ix)] = self.Cl.flatten()
            hf['data/Cl_'+str(ix)].attrs['shape'] = self.Cl.shape
            hf['data/Bs_'+str(ix)] = B_list[ix].flatten()
            hf['data/Bs_'+str(ix)].attrs['shape'] = B_list[ix].shape        
            hf['data/Cr_'+str(ix)] = self.Cr.flatten()
            hf['data/Cr_'+str(ix)].attrs['shape'] = self.Cr.shape
        hf['data/sigma'] = self.sigma

        hf.close()

    def growby2(self):
        newAs, newBs, initpsi = sadd.init_newpsi(self.sigma, self.As, self.Bs)
        newCl = contr.contractL(self.Cl, newAs, self.M)
        newCr = contr.contractR(self.Cr, newBs, self.M)

        H2 = np.einsum('abde,cdfg->abcefg', newCl, newCr)
        dimH2 = newCl.shape[1]**2
        H2 = np.reshape(H2, (dimH2, dimH2))
        H2s = scipy.sparse.csr_matrix(H2)

        if not np.allclose(H2, H2.T.conj()):
            warnings.warn(str(self.currL)+': H2 not Hermitian')

        initpsi = np.reshape(initpsi, (dimH2, 1))
        newpsi = scipy.sparse.linalg.eigsh(H2s, k=1, which='SA', v0=initpsi)[1]
        newpsi = np.reshape(newpsi, (newCl.shape[1], newCl.shape[1]))
        u, s, vdag = np.linalg.svd(newpsi)

        # update
        self.currL += 2
        self.As = np.einsum('abc,cd->abd', newAs, u)
        self.Bs = np.einsum('ac,bcd->bad', vdag, newBs)
        self.sigma = s

        self.Cl = np.einsum('abcd,be->aecd', newCl, u.conj())
        self.Cl = np.einsum('abcd,de->abce', self.Cl, u)
        self.Cr = np.einsum('ab,bcde->acde', vdag.conj(), newCr)
        self.Cr = np.einsum('ad,bcde->bcae', vdag, self.Cr)

        # save data
        hf = h5py.File(self.dataname, 'r+')
        ix = self.currL // 2 - 1
        hf['data/As_'+str(ix)] = self.As.flatten()
        hf['data/As_'+str(ix)].attrs['shape'] = self.As.shape
        hf['data/Cl_'+str(ix)] = self.Cl.flatten()
        hf['data/Cl_'+str(ix)].attrs['shape'] = self.Cl.shape
        hf['data/Bs_'+str(ix)] = self.Bs.flatten()
        hf['data/Bs_'+str(ix)].attrs['shape'] = self.Bs.shape        
        hf['data/Cr_'+str(ix)] = self.Cr.flatten()
        hf['data/Cr_'+str(ix)].attrs['shape'] = self.Cr.shape

        del hf['parameters/currL']
        hf['parameters/currL'] = self.currL
        del hf['data/sigma']
        hf['data/sigma'] = self.sigma

        hf.close()

    def grow_to_target(self):
        while self.currL < targetL:
            self.growby2()
        return 0

if __name__ == '__main__':
    bondD = 50
    targetL = 50
    solver = idmrg(bondD, targetL)
    solver.growby2()
    #solver.grow_to_target()
