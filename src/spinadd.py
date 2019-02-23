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
    u, sr, vdag = np.linalg.svd(mat_prod, full_matrices=False)
    newAs = np.reshape(u, (2, u.shape[0]//2, u.shape[1]))
    lambdaR = np.diag(sr).dot(vdag)
    return newAs, sr, lambdaR

def obtain_newBs(sigma, As):
    dim = len(sigma)
    Asfilled = fillzeros(As, dim)
    mat_prod = np.einsum('abc,cd->bad', Asfilled, np.diag(sigma))
    mat_prod = np.reshape(mat_prod, (mat_prod.shape[0], mat_prod.shape[1]*mat_prod.shape[2]))
    u, sl, vdag = np.linalg.svd(mat_prod, full_matrices=False)
    newBs = np.reshape(vdag, (vdag.shape[0], 2, vdag.shape[1]//2))
    newBs = np.swapaxes(newBs, 0, 1)
    lambdaL = u.dot(np.diag(sl))
    return newBs, sl, lambdaL

def init_newpsi(sigma, As, Bs):
    newAs, sr, lambdaR = obtain_newAs(sigma, Bs)
    newBs, sl, lambdaL = obtain_newBs(sigma, As)

    calc_rvs = np.frompyfunc(rvs, 1, 1)
    sr_rvsd = calc_rvs(sr)
    newpsi = lambdaR.dot(np.diag(sr_rvsd)).dot(lambdaL)

    return newAs, newBs, newpsi
