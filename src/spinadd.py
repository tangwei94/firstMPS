import numpy as np 
from numpy import linalg
import ed
from canonical import canonical
import sys

def obtain_newAs(sigma, Bs_list):
    mat = np.diag(sigma)
    newAs_list = []
    for ix in range(len(Bs_list)):
        mat_prod = np.einsum('bc,acd->abd', mat, Bs_list[ix])
        mat_prod = np.reshape(mat_prod, (mat_prod.shape[0]*mat_prod.shape[1], mat_prod.shape[2]))
        import pdb; pdb.set_trace()
        if ix < len(Bs_list) - 1:
            q, mat = np.linalg.qr(mat_prod)
        else:
            q, s, vdag = np.linalg.svd(mat_prod)
        newAs_list.append(np.reshape(q, (2, q.shape[0]//2, q.shape[1])))
    return newAs_list, s, vdag

def obtain_newBs(sigma, As_list):
    mat = np.diag(sigma)
    newBs_list = []
    for ix in range(len(As_list)):
        mat_prod = np.einsum('abc,cd->bad', As_list[ix], mat)
        mat_prod = np.reshape(mat_prod, (mat_prod.shape[0], mat_prod.shape[1]*mat_prod.shape[2]))
        if ix < len(As_list) - 1:
            mat_prod_dag = mat_prod.T.conj()
            q, mat = np.linalg.qr(mat_prod_dag)
            mat, q = mat.T.conj(), q.T.conj()
        else:
            u, s, q = np.linalg.svd(mat_prod)
        newBs_list.append(np.reshape(q, (2, q.shape[0]//2, q.shape[1])))
    return newBs_list, s, u

def fillzeros(Ms, dim):
    spin, shapel, shaper = Ms.shape
    if spin != 2 or shapel > dim or shaper > dim:
        sys.exit('fillzeros: dimension mismatch')
    if shapel < dim:
        Ms = np.append(Ms, np.zeros(2, dim - shapel, shaper), axis=1)
    if shaper < dim:
        Ms = np.append(Ms, np.zeros(2, dim, dim - shaper), axis=2)
    return Ms

def init_newpsi(sigma, As_list, Bs_list):
    newAs_list, sr, lambdaR = obtain_newAs(sigma, Bs_list)
    newBs_list, sl, lambdaL = obtain_newBs(sigma, As_list)

    if not np.allclose(sr, sl):
        sys.exit('sr, sl not equal to each other')
    sigma1 = sr

    maxdim = len(sigma)
    fillzero_to_list = np.frompyfunc(lambda Ms: fillzeros(Ms, maxdim), 1, 1)
    newAs_list = fillzero_to_list(newAs_list)
    newBs_list = fillzero_to_list(newBs_list)

    def rvs(x):
        if x > 1e-8: 
            return 1/x
        else: 
            return 0
    calc_rvs = np.frompyfunc(rvs, 1, 1)
    sigma1_rvsd = calc_rvs(sigma1)
    sigma1_rvsd = np.append(sigma1_rvsd, np.zeros(maxdim - len(sigma1)))
    
    lambdas = fillzeros(np.array([lambdaR, lambdaL]), maxdim)
    newpsi = lambdas[0].dot(sigma1_rvsd).dot(lambdas[1])

    return newAs_list, newBs_list, newpsi

if __name__ == '__main__':
    L = 8
    H = ed.xy_hamilt(L)
    E, psi = ed.ground_state(H)
    A_list, B_list, sigma = canonical(psi, L)
    sigma1 = obtain_newAs(sigma, [B_list[L//2-1], B_list[L//2-2]])[1]
    sigma1_left = obtain_newBs(sigma, [A_list[L//2-1], A_list[L//2-2]])[1]
    print(np.allclose(sigma1, sigma1_left))

    #newpsi = init_newpsi(sigma, A_list[L//2-1], B_list[L//2-1])
