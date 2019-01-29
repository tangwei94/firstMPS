import numpy as np 
from numpy import linalg
import scipy
from scipy import sparse
from scipy.sparse import linalg
import h5py
import contraction as contr
import idmrg
import mpo
import datafile as df
import warnings

def singlesite_hamilt(Cl, Cr, M, Ms):
    # construct the H mat
    Cl = np.reshape(Cl, (Cl.shape[1], Cl.shape[2], Cl.shape[3]))
    Cr = np.reshape(Cr, (Cr.shape[0], Cr.shape[1], Cr.shape[2]))
    Hss = np.einsum('ceg,abef,dfh->acdbgh', Cl, M, Cr)
    Hss = np.reshape(Hss, (Hss.shape[0]*Hss.shape[1]*Hss.shape[2], Hss.shape[3]*Hss.shape[4]*Hss.shape[5]))
    if not np.allclose(Hss, Hss.T.conj()):
        warnings.warn('singlesite_hamilt: Hss not hermitian')

    # solve
    Hss = scipy.sparse.csr_matrix(Hss)
    psi = Ms.flatten()
    E, psi = scipy.sparse.linalg.eigsh(Hss, k=1, which='SA', v0=psi)
    newMs = np.reshape(psi, Ms.shape)

    return E[0], newMs

def mpsE(Cl, Cr, Ms, M):
    Cl = contr.contractL(Cl, Ms, M)
    Cl, Cr = Cl.flatten(), Cr.flatten() 
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

def indexR(L, ix):
    return L - ix - 1

class dmrg3s(object):
    def __init__(self, L, D):
        self.L = L
        self.D = D
        self.dataname = 'data/xy_dmrg_D'+str(self.D)+'.h5'

        self.M = mpo.xy_mpo()
        self.alpha = 0.1
        self.E = 9.99e9
        self.reducedE = 9.99e9

        idmrgsol = idmrg.idmrg(self.D, self.L)
        idmrgsol.grow_to_target()

        self.ix = L // 2
        
        hf = h5py.File(self.dataname, 'r+')
        Ms = df.readarr(hf, 'data/Bs_'+ str(indexR(self.L, self.ix)))
        self.Ms = np.einsum('bc,acd->abd', np.diag(hf['data/sigma']), Ms)
        df.savearr(hf, 'data/Cl_-1', np.reshape([1,0,0,0], (1,1,4,1)))
        df.savearr(hf, 'data/Cr_-1', np.reshape([0,0,0,1], (1,4,1,1)))
        del hf['data/Bs_'+str(indexR(self.L, self.ix))]
        del hf['data/Cr_'+str(indexR(self.L, self.ix))]
        hf.close()

    def adjust_alpha(self, E):
        expectedE_upbound = self.E - self.reducedE * 0.6
        expectedE_lowbound = self.E - self.reducedE * 0.8
        if E > expectedE_upbound:
            self.alpha /= 1.2
        elif E < expectedE_lowbound:
            self.alpha *= 1.2

    def moveR(self):
        # read As, Bs, Cl, Cr
        hf = h5py.File(self.dataname, 'r')
        Bs = df.readarr(hf, 'data/Bs_'+ str(indexR(self.L, self.ix+1)))
        Cl = df.readarr(hf, 'data/Cl_'+ str(self.ix-1))
        Cr = df.readarr(hf, 'data/Cr_'+ str(indexR(self.L, self.ix+1)))
        hf.close()

        # energy of the present state
        self.E = mpsE(Cl, Cr, self.Ms, self.M)
        self.adjust_alpha(self.E)

        # construt single-site H and solve
        E, Ms = singlesite_hamilt(Cl, Cr, self.M, self.Ms)
        self.reducedE = self.E - E

        # subspace expansion
        P = self.alpha * obtainPR(Cl, Ms, self.M)
        Ms_tilde = np.append(Ms, P, axis=2)
        zeros = np.zeros((2, P.shape[2], Bs.shape[2]))
        Bs_tilde = np.append(Bs, zeros, axis=1)

        # truncation
        Ms_tilde = np.reshape(Ms_tilde, (Ms_tilde.shape[0]*Ms_tilde.shape[1], Ms_tilde.shape[2]))
        q, r = np.linalg.qr(Ms_tilde)
        Dr = self.Ms.shape[2]
        if q.shape[1] > Dr:
            q = np.compress([1]*Dr, q, axis=1)
            r = np.compress([1]*Dr, r, axis=0)
        
        # new As,Cl and new Ms
        newAs = np.reshape(q, (2, q.shape[0]//2, q.shape[1]))
        newCl = contr.contractL(Cl, newAs, self.M)
        self.Ms = np.einsum('bc,acd->abd', r, Bs_tilde)

        # update datafile
        hf = h5py.File(self.dataname, 'r+')
        df.savearr(hf, 'data/As_'+str(self.ix), newAs)
        df.savearr(hf, 'data/Cl_'+str(self.ix), newCl)
        self.ix += 1
        del hf['data/Bs_'+str(indexR(self.L, self.ix))]
        del hf['data/Cr_'+str(indexR(self.L, self.ix))]
        hf.close()

    def moveL(self):
        # read As, Bs, Cl, Cr
        hf = h5py.File(self.dataname, 'r')
        As = df.readarr(hf, 'data/As_'+ str(self.ix-1))
        Cl = df.readarr(hf, 'data/Cl_'+ str(self.ix-1))
        Cr = df.readarr(hf, 'data/Cr_'+ str(indexR(self.L, self.ix+1)))
        hf.close()

        # energy of the present state
        newE = mpsE(Cl, Cr, self.Ms, self.M)
        self.adjust_alpha(newE)
        self.E = newE

        # construt single-site H and solve
        E, Ms = singlesite_hamilt(Cl, Cr, self.M, self.Ms)
        self.reducedE = self.E - E

        # subspace expansion
        P = self.alpha * obtainPL(Cr, Ms, self.M)
        Ms_tilde = np.append(Ms, P, axis=1)
        zeros = np.zeros((2, As.shape[1], P.shape[1]))
        As_tilde = np.append(As, zeros, axis=2)

        # truncation
        Ms_tilde = np.swapaxes(Ms_tilde, 0, 1)
        Ms_tilde = np.reshape(Ms_tilde, (Ms_tilde.shape[0], Ms_tilde.shape[1]*Ms_tilde.shape[2]))
        Ms_tilde_dag = Ms_tilde.T.conj()
        q, r = np.linalg.qr(Ms_tilde_dag)
        q, r = q.T.conj(), r.T.conj()
        Dl = self.Ms.shape[1]

        if q.shape[0] > Dl:
            q = np.compress([1]*Dl, q, axis=0)
            r = np.compress([1]*Dl, r, axis=1)

        # new Bs, Cr and new Ms
        newBs = np.reshape(q, (q.shape[0], 2, q.shape[1]//2))
        newBs = np.swapaxes(newBs, 0, 1)
        newCr = contr.contractR(Cr, newBs, self.M)
        self.Ms = np.einsum('abc,cd->abd', As_tilde, r)

        # update datafile
        hf = h5py.File(self.dataname, 'r+')
        df.savearr(hf, 'data/Bs_'+str(indexR(self.L, self.ix)), newBs)
        df.savearr(hf, 'data/Cr_'+str(indexR(self.L, self.ix)), newCr)
        self.ix -= 1
        del hf['data/As_'+str(self.ix)]
        del hf['data/Cl_'+str(self.ix)]
        hf.close()

if __name__ == '__main__':
    L = 100
    D = 24
    Egs = lambda L: -1*(np.sin((L//2+0.5)*np.pi/(L+1))/np.sin(0.5*np.pi/(L+1))/2-0.5)
    E_exact = Egs(L)
    solver = dmrg3s(L, D)

    takereal = lambda x: np.around(x.real, 6)
    takelog = lambda x: np.log10(np.abs(takereal(x)))
    for ix in range(L//2 - 1):
        solver.moveR()
        print(solver.ix, takereal(solver.E), takelog(E_exact-solver.E), takereal(solver.reducedE))

    def sweap():
        for ix in range(L - 1):
            solver.moveL()
            print(solver.ix, takereal(solver.E), takelog(E_exact-solver.E), takereal(solver.reducedE))
        for ix in range(L - 1):
            solver.moveR()
            print(solver.ix, takereal(solver.E), takelog(E_exact-solver.E), takereal(solver.reducedE))
    for ix in range(5):
        sweap()


        




