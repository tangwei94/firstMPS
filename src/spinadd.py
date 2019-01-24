import numpy as np 
from numpy import linalg
import ed
from canonical import canonical
import contraction as contr
import warnings

def rvs(x):
    if x > 1e-8: 
        return 1/x
    else: 
        return 0

def fillzeros(Ms, dim):
    spin, shapel, shaper = Ms.shape
    if spin != 2 or shapel > dim or shaper > dim:
        warnings.warn('fillzeros: dimension mismatch')
    if shapel < dim:
        Ms = np.append(Ms, np.zeros((2, dim - shapel, shaper)), axis=1)
    if shaper < dim:
        Ms = np.append(Ms, np.zeros((2, dim, dim - shaper)), axis=2)
    return Ms

def obtain_newAs(sigma, Bs):
    dim = len(sigma)
    Bsfilled = fillzeros(Bs, dim)
    mat_prod = np.einsum('bc,acd->abd', np.diag(sigma), Bsfilled)
    mat_prod = np.reshape(mat_prod, (mat_prod.shape[0]*mat_prod.shape[1], mat_prod.shape[2]))
    u, sr, vdag = np.linalg.svd(mat_prod)
    newAs = np.reshape(u, (2, u.shape[0]//2, u.shape[1]))
    lambdaR = np.diag(sr).dot(vdag)
    return newAs, sr, lambdaR

def obtain_newBs(sigma, As):
    dim = len(sigma)
    Asfilled = fillzeros(As, dim)
    mat_prod = np.einsum('abc,cd->bad', Asfilled, np.diag(sigma))
    mat_prod = np.reshape(mat_prod, (mat_prod.shape[0], mat_prod.shape[1]*mat_prod.shape[2]))
    u, sl, vdag = np.linalg.svd(mat_prod)
    newBs = np.reshape(vdag, (vdag.shape[0], 2, vdag.shape[1]//2))
    newBs = np.swapaxes(newBs, 0, 1)
    lambdaL = u.dot(np.diag(sl))
    return newBs, sl, lambdaL

def init_newpsi(sigma, As, Bs):
    newAs, sr, lambdaR = obtain_newAs(sigma, Bs)
    newBs, sl, lambdaL = obtain_newBs(sigma, As)

    if not np.allclose(sl, sr, atol=1.e-5):
        warnings.warn('init_newpsi: sl != sr')

    calc_rvs = np.frompyfunc(rvs, 1, 1)
    sr_rvsd = calc_rvs(sr)
    newpsi = lambdaR.dot(np.diag(sr_rvsd)).dot(lambdaL)

    return newAs, newBs, newpsi

if __name__ == '__main__':
    L = 10
    H = ed.xy_hamilt(L)
    E, psi = ed.ground_state(H)
    A_list, B_list, sigma = canonical(psi, L)

    newAs, newBs, newpsi = init_newpsi(sigma, A_list[L//2-1], B_list[L//2-1])
    A_list += [newAs]
    B_list += [newBs]

    if contr.check_Lnorm(A_list):
        print('left normalization condition: test passed')
    else:
        print('left normalization condition: test failed')
    if contr.check_Rnorm(B_list):
        print('right normalization condition: test passed')
    else:
        print('right normalization condition: test failed')
