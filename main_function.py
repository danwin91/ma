import numpy as np
import sigmoid
import tanh
from tensor_util import tensor2, tensor3


valid_function_names = ["sigmoid", "tanh", "exp", "constant"]



def set_g(g_name):
    global g,dg,ddg,dddg
 
    assert g_name in valid_function_names, "[main_function.py] Unknown function"
 
    print("[main_function.py] Found {}, setting g={}".format(g_name, g_name))
    if g_name == "sigmoid":
        g,dg,ddg,dddg, = sigmoid.g, sigmoid.dg, sigmoid.ddg, sigmoid.dddg
    elif g_name == "tanh":
        g,dg,ddg,dddg, = tanh.g, tanh.dg, tanh.ddg, tanh.dddg
    elif g_name == "exp":
        g = dg = ddg = dddg = np.exp
    

def f_inner(x,A,foo):
    return foo(A.T.dot(x))

def f(x,A,B):
    return np.sum(g(B.T.dot(g(A.T.dot(x)))))

def df(X,A,B):
    ret = []
    for x in X:
        #V = A diag(dg(A^T x)) B in R^dxm_1
        V = A.dot(np.diag(dg(A.T.dot(x))).dot(B))
        inner = f_inner(x,A,g)
        dH = dg(B.T.dot(g(A.T.dot(x))))
        ret.append(V.dot(dH))
    return ret
"""
def ddf(X,a,b):
    ret = []
    samples = len(X)
    d, m = a.shape
    for x in X:
        right = np.dot(a, b[:,0]*f_inner(x,a,dg))
        right2 = np.tensordot(right,right,axes=0)
        term1 = ddg(np.dot(b[:,0], f_inner(x,a,g)))*right2
        term2 = dg(np.dot(b[:,0],f_inner(x,a,g)))
        temp = np.zeros(shape = (d,d))
        for i in range(m):
            temp += b[i,0]*ddg(np.dot(a[:,i],x))*np.tensordot(a[:,i],a[:,i],axes=0)
        r = term1 + term2*temp
        ret.append(r)
    
    
    
    return ret
"""
def ddf(X,A,B):
    ret = []

    for x in X:
        #term1 containing v_e:
        V = A.dot(np.diag(dg(A.T.dot(x))).dot(B))
        term1 = V.dot(np.diag(ddg(B.T.dot(g(A.T.dot(x))))).dot(V.T))
        
        #factor, -> diagonal BH*g''(A^Tx)
        dH = dg(B.T.dot(g(A.T.dot(x))))
        factor = B.dot(dH)*ddg(A.T.dot(x))
        term2 = A.dot(np.diag(factor).dot(A.T))
        ret.append(term1+term2)
    
    return ret



def dddf_single(x, A,B):
    #consists of 4m+m_1 terms seperated into 3 terms pure, semi, mixed
    d,m = A.shape
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
