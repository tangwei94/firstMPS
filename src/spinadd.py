import numpy as np 
from numpy import linalg
import ed
from canonical import canonical

def obtain_sigma1(sigma, Bs):
    mat_prod = []
    sigma = np.diag(sigma)
    mat_prod.append(sigma.dot(Bs[0]))
    mat_prod.append(sigma.dot(Bs[1]))
    mat_prod = np.asarray(mat_prod)
    shape0 = mat_prod.shape
    mat_prod = np.reshape(mat_prod, (shape0[0]*shape0[1], shape0[2]))
    sigma1 = np.linalg.svd(mat_prod, compute_uv=False)

    return sigma1

def obtain_sigma1_left(sigma, As): # for test
    mat_prod = []
    sigma = np.diag(sigma)
    mat_prod.append(As[0].dot(sigma))
    mat_prod.append(As[1].dot(sigma))
    mat_prod = np.asarray(mat_prod)
    mat_prod = np.swapaxes(mat_prod, 0, 1) # careful
    shape0 = mat_prod.shape
    mat_prod = np.reshape(mat_prod, (shape0[0], shape0[1]*shape0[2]))
    sigma1 = np.linalg.svd(mat_prod, compute_uv=False)
    return sigma1

def init_newpsi(sigma, As, Bs): # not tested
    
    sigma1 = obtain_sigma1(sigma, Bs)
    sigma = np.diag(sigma)

    maxdim = Bs[0].shape[0]

    matprod_l = np.asarray([sigma.dot(Bs[0]), sigma.dot(Bs[1])])
    shape_l = matprod_l.shape
    matprod_l = np.reshape(matprod_l, (shape_l[0]*shape_l[1], shape_l[2]))
    if (maxdim > matprod_l.shape[1]):
        diff_shape = (matprod_l.shape[0], maxdim - matprod_l.shape[1])
        matprod_l = np.append(matprod_l, np.zeros(diff_shape, dtype=np.complex), axis=1)
    u, lambdaR = np.linalg.qr(matprod_l)
    newAs = np.reshape(u, (2, u.shape[0]//2, u.shape[1]))

    matprod_r = np.asarray([As[0].dot(sigma), As[1].dot(sigma)])
    matprod_r = np.swapaxes(matprod_r, 0, 1)
    shape_r = matprod_r.shape
    matprod_r = np.reshape(matprod_r, (shape_r[0], shape_r[1]*shape_r[2]))
    matprod_r_dag = matprod_r.T.conj()
    if (maxdim > matprod_r_dag.shape[1]):
        diff_shape = (matprod_r_dag.shape[0], maxdim - matprod_r_dag.shape[1])
        matprod_r_dag = np.append(matprod_r_dag, np.zeros(diff_shape, dtype=np.complex), axis=1)
    v, lambdaL_dag = np.linalg.qr(matprod_r_dag)
    vdag, lambdaL = v.T.conj(), lambdaL_dag.T.conj()
    newBs = np.reshape(vdag, (vdag.shape[0], 2, vdag.shape[1]//2))
    newBs = np.swapaxes(newBs, 0, 1)

    def rvs(x):
        if x > 1e-8: 
            return 1/x
        else: 
            return 0
    calc_rvs = np.frompyfunc(rvs, 1, 1)
    sigma1_rvsd = calc_rvs(sigma1)
    if len(sigma1_rvsd) < maxdim:
        sigma1_rvsd = np.append(sigma1_rvsd, np.zeros(maxdim - len(sigma1_rvsd)))
    sigma1_rvsd = np.diag(calc_rvs(sigma1_rvsd))

    newpsi = lambdaR.dot(sigma1_rvsd).dot(lambdaL)

    return newAs, newBs, newpsi

if __name__ == '__main__':
    L = 6
    H = ed.xy_hamilt(L)
    E, psi = ed.ground_state(H)
    A_list, B_list, sigma = canonical(psi, L)
    sigma1 = obtain_sigma1(sigma, B_list[L//2-1])
    sigma1_left = obtain_sigma1_left(sigma, A_list[L//2-1])
    print(np.allclose(sigma1, sigma1_left))

    newpsi = init_newpsi(sigma, A_list[L//2-1], B_list[L//2-1])
