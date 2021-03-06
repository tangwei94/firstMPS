import numpy as np 
import ed, mpo
from canonical import canonical

def contractL(Cl, As, M):
    # contract from the left
    El = np.einsum('ahi,abjk->bhijk', As.conj(), M)
    El = np.einsum('bhijk,blm->hjlikm', El, As)
    El = np.reshape(El, (El.shape[0]*El.shape[1]*El.shape[2], El.shape[3]*El.shape[4]*El.shape[5]))
    Clnew = Cl.flatten()
    Clnew = np.reshape(Clnew, (1, len(Clnew)))
    Clnew = Clnew.dot(El)
    newshape = (1, As[0].shape[1], M[0][0].shape[0], As[0].shape[1])
    Clnew = np.reshape(Clnew, newshape)
    return Clnew

def contractR(Cr, Bs, M):
    # contract from the right
    Er = np.einsum('ahi,abjk->bhijk', Bs.conj(), M)
    Er = np.einsum('bhijk,blm->hjlikm', Er, Bs)
    Er = np.reshape(Er, (Er.shape[0]*Er.shape[1]*Er.shape[2], Er.shape[3]*Er.shape[4]*Er.shape[5]))
    Crnew = Cr.flatten()
    Crnew = np.reshape(Crnew, (len(Crnew), 1))
    Crnew = Er.dot(Crnew)
    newshape = (Bs[0].shape[0], M[0][0].shape[1], Bs[0].shape[0], 1)
    Crnew = np.reshape(Crnew, newshape)
    return Crnew

#### tests
def check_Lnorm(A_list):
    ispassed = True

    I_mpo = np.reshape([1,0,0,1], (2,2,1,1))
    Cl = np.reshape([1], (1,1,1,1))
    for ix in range(len(A_list)):
        Cl = contractL(Cl, A_list[ix], I_mpo)
        if not np.allclose(np.reshape(Cl, (Cl.shape[1], Cl.shape[3])), np.identity(Cl.shape[1])):
            ispassed = False

    return ispassed

def check_Rnorm(B_list):
    ispassed = True
    
    I_mpo = np.reshape([1,0,0,1], (2,2,1,1))
    Cr = np.reshape([1], (1,1,1,1))
    for ix in range(len(B_list)):
        Cr = contractR(Cr, B_list[ix], I_mpo)
        if not np.allclose(np.reshape(Cr, (Cr.shape[0], Cr.shape[2])), np.identity(Cr.shape[0])):
            ispassed = False
    
    return ispassed
