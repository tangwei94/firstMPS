import numpy as np 

def contractL(Cl, As, M):
    # contract from the left
    El = np.einsum('ahi,abjk,blm->hjl,ikm', As.conj, M, As)
    El = np.reshape(El, (El.shape[0]*El.shape[1]*El.shape[2], El.shape[3]*El.shape[4]*El.shape[5]))
    Clnew = np.reshape(Cl, (1, Cl.shape[0]*Cl.shape[1]*Cl.shape[2]))
    Clnew = Clnew.dot(El)
    Clnew = np.reshape(Clnew, Cl.shape)
    return Clnew

def contractR(Cr, Bs, M):
    # contract from the right
    Er = np.einsum('ahi,abjk,blm->hjl,ikm', Bs.conj(), M, Bs)
    Er = np.reshape(Er, (Er.shape[0]*Er.shape[1]*Er.shape[2], Er.shape[3]*Er.shape[4]*Er.shape[5]))
    Crnew = np.reshape(Cr, (Cr.shape[0]*Cr.shape[1]*Cr.shape[2], 1))
    Crnew = Er.dot(Crnew)
    Crnew = np.reshape(Crnew, Cr.shape)
    return Crnew

if __name__ == '__main__':
    print('hello')
