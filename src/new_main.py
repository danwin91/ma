import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#own modules
from decomp_algorithm import run_algorithm, algo1
#from tanh import g,dg,ddg,dddg
import myplots
from tensor_util import *
import json
import time
g_name = "sigmoid"


m_1 = 5
m = 10
d = 30
m_x = 5000
res_2, res_3, data, ddf = run_algorithm(m,m_1,d,m_x, g_name = g_name,  symm = False, verbose = True, mode = [2])
A, B, X = data
if res_2:
    L_2, D_2, V_2 = res_2

k = m #m or m_1+m the number of basisvectors of L_2 used

v = np.reshape(X[0], (d,d))

A_tensor = np.array([np.reshape(tensor2(A[:, i], A[:,i]) , d**2) for i in range(A.shape[1])]).T
Ab_tensor = np.array([np.reshape(tensor2(A.dot(B)[:, i], A.dot(B)[:,i]), d**2) for i in range(m_1)]).T
resu=np.reshape(algo1(v,L_2[:,:k], gamma=1.5,n = 30), dd**2)
print(A_tensor.T.dot(resu))
print("----")
print(Ab_tensor.T.dot(resu))
