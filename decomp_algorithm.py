import numpy as np
import tensor_util
from scipy.stats import ortho_group
from main_function import *
import time

def perm_matrix(m):
    P = np.identity(m)
    P = np.random.permutation(P)#permutes only the rows
    return P, np.linalg.inv(P)


def uniformSphere(d, m_x):
    x = np.random.normal(size=(m_x,d))
    for i in range(m_x):
        x[i] = x[i] / np.linalg.norm(x[i])
    return x


#Algorithm FOSV:
def algo1(X, Te, gamma = 1.1, n = 10):
    dd = X.shape[0]
    for i in range(n):
        U, D, V = np.linalg.svd(X, full_matrices=True)
        D[0] *= gamma
        D=D/np.linalg.norm(D)
        X = U.dot(np.diag(D).dot(V))
        X = np.reshape(X, dd**2)
        #Project on space spanned by T:
        X=np.reshape(Te.dot(Te.T.dot(X)), (dd,dd))
    return X

def decomp_svd(tensors, symm = True, randomize=False):
    """tbd.

    Parameters
    ----------
    X : matrix
        m_x samples \in R^m
    A : matrix
        m orthonormal vectors \in R^m
    B : matrix
        m_1 orthonormal vectors \in R^m
    randomize: bool
        if true performs a random unfolding
    Returns
    -------
    alot

    """
    mode = len(tensors[0].shape)
    d = tensors[0].shape[0]

    if symm:
        unfolds = [tensor_util.vectorize_symm_tensor(t) for t in tensors]
    else:
        unfolds = [np.reshape(t, (d**mode,)) for t in tensors]
    
    if randomize:
        P,Q = perm_matrix(d**mode)
        unfolds = [np.dot(P,v) for v in unfolds]
        
    M = np.transpose(np.array(unfolds))
    U,D,V = np.linalg.svd(M, full_matrices=False)
    if randomize: 
        U = np.dot(Q,U)
    return U, D, V



def run_algorithm(m,m_1,d,m_x,g_name, symm = True, verbose = False, mode = [2,3]):

    if verbose:
        print("[run_alg] Trying to set g = {}".format(g_name))
    set_g(g_name)
    if verbose:
        start  = time.time()
        print("[run_alg] Creating data...")
    #Create data
    #Setting parameters, and creating data
    X = uniformSphere(d, m_x)
    A = ortho_group.rvs(dim = d)[:m]# m orthogonal vectors with dim m, a_1, ..., a_m
    #A = np.identity(d)[:,:m]
    A = np.transpose(A)
    B = ortho_group.rvs(dim = m)[:m_1]#m_1 orthogonal vectors with dimension m b_1, ..., b_m_1
    #B = np.identity(m)[:,:m_1]
    B = np.transpose(B)

    assert mode == [2,3] or mode == [2] or mode == [3], "invalid mode configuration"
    if 2 not in mode:
        ret_2 = None
    if 3 not in mode:
        ret_3 = None

    if verbose:
        tmp1 = time.time()
        print("[run_alg] Finished creating data, time elapsed {:.2f}s".format(tmp1 - start))
    
    if 2 in mode: 
        if verbose:
            print("[run_alg] Calculating second derivative...")
        
        ddf_values = ddf(X,A,B)
        
        if verbose:
            tmp2 = time.time()
            print("[run_alg] Finished second derivative, time elapsed {:.2f}s".format(tmp2 - tmp1))
            print("[run_alg] Decomposing M_2...")
        
        U_2, D_2, V_2 = decomp_svd(ddf_values, symm = symm)
        ret_2 = [U_2, D_2, V_2]
        if verbose:
            tmp1 = time.time()
            print("[run_alg] Finished decomposition of M_2, time elapsed {:.2f}s".format(tmp1 - tmp2))

    if 3 in mode: 
        if verbose:
            print("[run_alg] Calculating third derivative...")
        dddf_values = dddf(X,A,B)

        if verbose:
            tmp2 = time.time()
            print("[run_alg] Finished third derivative, time elapsed {:.2f}s".format(tmp2 - tmp1))
            print("[run_alg] Decomposing M_3...")

        U_3, D_3, V_3 = decomp_svd(dddf_values, symm = symm)
        ret_3 = [U_3, D_3, V_3]

        if verbose:
            tmp1 = time.time()
            print("[run_alg] Finished decomposition of M_3, time elapsed {:.2f}s".format(tmp1 - tmp2))
            
       
    if verbose:   
        print("[run_alg] Returning, hole execution time: {:.2f}s".format(tmp1 - start))


    return ret_2, ret_3, [A,B,X], ddf_values

