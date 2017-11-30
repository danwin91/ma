import numpy as np
from sigmoid import g,dg,ddg,dddg
from tensor_util import tensor2, tensor3

def f_inner(x,A,foo):
    m = len(x)
    v = [foo(np.dot(x,A[:, i])) for i in range(m)]
    return np.array(v)

def f(x,a,b):
    return g(np.dot(b[:,0],f_inner(x,a,g)))

def df(X,a,b):
    ret = []
    for x in X: 
        right = np.dot(A, b[0]*f_inner(x,a,dg))
        ret.append(dg(np.dot(b[:,0],f_inner(x,a,g)))*right)
    return ret

def ddf(X,a,b):
    ret = []
    samples = len(X)
    m = len(X[0])
    for x in X:
        right = np.dot(a, b[:,0]*f_inner(x,a,dg))
        right2 = np.tensordot(right,right,axes=0)
        term1 = ddg(np.dot(b[:,0], f_inner(x,a,g)))*right2
        term2 = dg(np.dot(b[:,0],f_inner(x,a,g)))
        temp = np.zeros(shape = (m,m))
        for i in range(m):
            temp += b[i,0]*ddg(np.dot(a[:,i],x))*np.tensordot(a[:,i],a[:,i],axes=0)
        r = term1 + term2*temp
        ret.append(r)
    return ret

def dddf_single(x, A,B):
    #consists of 4m+m_1 terms seperated into 3 terms pure, semi, mixed
    m = len(x)
    pure = 0
    for i in range(m):
        pure += B[i, 0]*dddg(np.dot(A[:,i],x))*tensor3(A[:,i],A[:,i],A[:,i])
    H_pure = dg(np.dot(B[:,0], f_inner(x, A, g)))
    pure *= H_pure
    
    v = np.dot(A, f_inner(x, A, dg)*B[:,0])
    
    #mixed
    mixed = tensor3(v,v,v)
    H_mixed = dddg(np.dot(B[:,0], f_inner(x,A,g)))
    mixed *= H_mixed
    
    #semi
    semi = 0
    for i in range(m):
        semi += B[i,0]*ddg(np.dot(A[:,i],x))*(tensor3(v,A[:,i], A[:, i])+tensor3(A[:,i],v, A[:, i])+tensor3(A[:,i], A[:, i],v))
    H_semi = ddg(np.dot(B[:, 0], f_inner(x,A,g)))
    semi*=H_semi
    res = pure + mixed + semi
    info = [np.linalg.norm(pure),H_pure, np.linalg.norm(semi), H_semi, np.linalg.norm(mixed), H_mixed]
    return res, info

def dddf(X,A,B):
    ret = []
    info = []
    samples = len(X)
    for x in X:
        r , inf= dddf_single(x,A,B)
        ret.append(r)
        info.append(inf)
    return ret
