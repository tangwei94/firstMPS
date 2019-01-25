import numpy as np 
from numpy import linalg
import scipy
from scipy import sparse
from scipy.sparse import linalg
import h5py
import contraction as contr
import idmrg
import mpo
import warnings

def singlesite_hamilt(Cl, Cr, M):
    Cl = np.reshape(Cl, (Cl.shape[1], Cl.shape[2], Cl.shape[3]))
    Cr = np.reshape(Cr, (Cr.shape[0], Cr.shape[1], Cr.shape[2]))
    Hss = np.einsum('cdg, abef,dfh->acdbgh', Cl, M, Cr)
    if not np.allclose(Hss, Hss.T.conj()):
        warnings.warn('singlesite_hamilt: Hss not hermitian')

    return Hss

def mpsE(Cl, Cr, Ms, M):
    Cl = contr.contractL(Cl, Ms, M)
    Cl = np.reshape(Cl, (Cl.shape[1], Cl.shape[2], Cl.shape[3]))
    Cr = np.reshape(Cr, (Cr.shape[0], Cr.shape[1], Cr.shape[2]))
    return Cl.dot(Cr)

def obtainPR(Cl, Ms, M):
    Cl = np.reshape(Cl, (Cl.shape[1], Cl.shape[2]*Cl.shape[3]))
    MsM = np.einsum('abcd,bef->acedf', M, Ms)
    MsM = np.reshape(MsM, (MsM.shape[0], MsM.shape[1]*MsM.shape[2], MsM.shape[3]*MsM.shape[4]))
    Pr = np.einsum('bc,acd->abd', Cl, MsM)
    return Pr

def obtainPL(Cr, Ms, M):
    Cr = np.reshape(Cr, (Cr.shape[0], Cr.shape[1]*Cr.shape[2]))
    MsM = np.einsum('abcd,bef->acedf', M, Ms)
    MsM = np.reshape(MsM, (MsM.shape[0], MsM.shape[1]*MsM.shape[2], MsM.shape[3]*MsM.shape[4]))
    Pl = np.einsum('abd,cd->abc', MsM, Cr)
    return Pl

def readarr(hf, datasetname):
    arr_dat = hf[datasetname]
    arr_shape = hf[datasetname].attrs['shape']
    arr = np.reshape(arr_dat, arr_shape)
    return arr

def indexR(L, ix):
    return L - ix - 1

class dmrg3s(object):
    def __init__(self, L, D):
        self.L = L
        self.D = D
        self.dataname = 'data/xy_dmrg_D'+str(self.D)+'.h5'

        self.M = mpo.xy_mpo()
        self.alpha = 0.2

        idmrgsol = idmrg.idmrg(self.D, self.L)
        idmrgsol.grow_to_target()

        self.ix = L // 2
        
        hf = h5py.File(self.dataname, 'r')
        Ms = readarr(hf, 'data/Bs_'+ str(indexR(self.L, self.ix)))
        self.Ms = np.einsum('bc,acd->abd', np.diag(hf['data/sigma']), Ms)
        hf.close()

    def moveR(self):
        # read As, Bs, Cl, Cr
        hf = h5py.File(self.dataname, 'r')
        Bs = readarr(hf, 'data/Bs_'+ str(indexR(self.L, self.ix+1)))
        Cl = readarr(hf, 'data/Cl_'+ str(self.ix-1))
        Cr = readarr(hf, 'data/Cr_'+ str(indexR(self.L, self.ix+1)))
        hf.close()

        # construt Hss and eigensolve
        Hss = singlesite_hamilt(Cl, Cr, self.M)
        Hss = scipy.sparse.csr_matrix(Hss)
        psi = self.Ms.flatten()

        E, psi = scipy.sparse.linalg.eigsh(Hss, k=1, which='SC', v0=psi)
        Ms = np.reshape(psi, self.Ms.shape)

        # subspace expansion
        P = self.alpha * obtainPR(Cl, Ms, self.M)
        Ms_tilde = np.append(Ms, P, axis=2)
        zeros = np.zeros((P.shape[2], Bs.shape[2]))
        Bs_tilde = np.append(Bs, zeros, axis=1)

        # truncation
        Ms_tilde = np.reshape(Ms_tilde, (Ms_tilde.shape[0]*Ms_tilde.shape[1], Ms_tilde.shape[2]))
        q, r = np.linalg.qr(Ms_tilde)
        Dr = self.Ms.shape[2]
        if q.shape[1] > Dr:
            q = np.compress([1]*Dr, q, axis=1)
            r = np.compress([1]*Dr, r, axis=0)
        
        # new As,Cl and new Ms
        newAs = np.reshape(q, (2, q.shape[0]/2, q.shape[1]))
        newCl = contr.contractL(Cl, newAs, self.M)
        self.Ms = np.einsum('bc,acd->abd', r, Bs_tilde)
        self.ix += 1

        # update datafile
        hf = h5py.File(self.dataname, 'r+')
        hf['data/As_'+str(self.ix-1)] = newAs.flatten()
        hf['data/As_'+str(self.ix-1)].attrs['shape'] = newAs.shape
        hf['data/Cl_'+str(self.ix-1)] = newCl.flatten()
        hf['data/Cl_'+str(self.ix-1)].attrs['shape'] = newCl.shape
        del hf['data/Bs_'+str(indexR(self.L, self.ix))]
        del hf['data/Cr_'+str(indexR(self.L, self.ix))]
        hf.close()











        




