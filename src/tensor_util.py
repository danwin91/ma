import numpy as np 

def tensor2(a,b):
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


def tensor_to_vec(t, mode = 2):
    #not unique but we map to first element positive
    assert(mode == 2), "other modes not implemented yet"
    if mode == 2:
        #assume t = axa
        a = np.diag(np.sqrt(np.abs(t))) * np.sign(t[:,0])
    return a



def vectorize_symm_tensor(t):
    mode = len(t.shape)
    d = t.shape[0]
    if mode == 2:
        ret = np.zeros(int(d*(d+1)/2))
        ind = 0
        for i in range(d):
            for j in range(i+1):
                ret[ind] = t[j][i]
                ind += 1

    elif mode == 3:
        ret = np.zeros(int(d*(d+1)*(d+2)/6.0))
        ind = 0
        for i in range(d):
            for j in range(i+1):
                for k in range(j+1):
                    ret[ind] = t[k][j][i]
                    ind += 1

    return ret