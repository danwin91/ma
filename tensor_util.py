import numpy as np 



def tensor2(a,b):
    """Calculates the tensorproduct of two vectors
    Parameters
    ----------
    param1 : vector
        The first vector.
    param2 : vector
        The second vector.
    Returns
    -------
    2-mode tensor
        tensorproduct of a,b of size m x n
    """
    return np.tensordot(a,b, axes=0)


def tensor3(a, b, c):
    return np.tensordot(np.tensordot(a,b,axes=0), c, axes=0)



def tensor_distance_to_columns(vec,U,max_cols=np.infty, mode=2):
    assert(len(vec.shape) ==1),"first argument has to be a vector"
    m = len(vec)
    if mode == 2:
        t = tensor2(vec,vec)
    elif mode == 3:
        t = tensor3(vec,vec,vec)
    #Bad pratice -> make tensor -> vectorize ... 
    t_vec = np.reshape(t, m**mode)
    dist = []
    for i in range(np.min([max_cols, U.shape[0]])):
        d1 = np.linalg.norm(t_vec - U[:, i])
        d2 = np.linalg.norm(-t_vec - U[:, i])
        dist.append(np.min([d1,d2]))
    return dist