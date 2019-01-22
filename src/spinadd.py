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



if __name__ == '__main__':
    L = 6
    H = ed.xy_hamilt(L)
    E, psi = ed.ground_state(H)
    A_list, B_list, sigma = canonical(psi, L)
    sigma1 = obtain_sigma1(sigma, B_list[L//2-1])
    sigma1_left = obtain_sigma1_left(sigma, A_list[L//2-1])
    print(np.allclose(sigma1, sigma1_left))