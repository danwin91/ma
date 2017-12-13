#Standard imports
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
g_name = "tanh"


def main():
    results = {'m':[], 'm_1':[], 'd':[], 'm_x':[], 'data':[]}

    for i in range(0,5):
        m_1 = 2**i
        for j in range(i,i+3):
            m = np.max([2**j,2])
            for k in range(3):
                d = 10*k+m
                m_x = 10000
                print("Results for m_1 = {}, m = {}, d = {}, m_x = {}".format(m_1, m,d,m_x))
                r = run_sim(m, m_1, d, m_x)
                #results[str((m,m_1,d,m_x))] = r
                results['m'].append(m)
                results['m_1'].append(m_1)
                results['d'].append(d)
                results['m_x'].append(m_x)
                results['data'].append(r)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    df = pd.DataFrame(data = results)
    df.to_csv(g_name+timestr+'.csv')
    
    """
    with open(g_name+timestr+'.json', 'w') as outfile:
        json.dump(results, outfile)
    """
    



def run_sim(m,m_1,d,m_x=10000):
    #Data
    res_2, res_3, data, ddf = run_algorithm(m,m_1,d,m_x, g_name = g_name,  symm = False, verbose = False, mode = [2])
    if res_2:
        L_2, D_2, V_2 = res_2
    if res_3:    
        L_3, D_3, V_3 = res_3
    A,B,X = data



    results = []

    #Tensors of A, AB
    A_tensor = [np.reshape(tensor2(A[:, i], A[:,i]) , d**2) for i in range(A.shape[1])]
    Ab_tensor = [np.reshape(tensor2(A.dot(B)[:, i], A.dot(B)[:,i]), d**2) for i in range(m_1)]
    #second  deriv. results
    if res_2:
        print("Projection T_[m]:")
        A_big = np.array(A_tensor).T
        Ab_big = np.array(Ab_tensor).T
        proj_dist_m = [np.linalg.norm(L_2[:,:m].dot(L_2[:,:m].T).dot(A_big[:,i]) - A_big[:,i]) for i in range(m)]
        max_proj_dist_A_m = np.max(proj_dist_m)
        min_proj_dist_A_m = np.min(proj_dist_m)
        results.extend([max_proj_dist_A_m,min_proj_dist_A_m])
        print("Maximal distance P_T(A)={}".format(max_proj_dist_A_m))        
        print("Minimal distance P_T(A)={}".format(min_proj_dist_A_m))

        
        projection_dist_Ab_m = [np.linalg.norm(L_2[:,:m].dot(L_2[:,:m].T).dot(Ab_big[:,i]) - Ab_big[:,i]) for i in range(m_1)]
        max_proj_dist_Ab_m = np.max(projection_dist_Ab_m)
        min_proj_dist_Ab_m = np.min(projection_dist_Ab_m)
        results.extend([max_proj_dist_Ab_m,min_proj_dist_Ab_m])    
        print("Maximal distance P_T(Ab)={}".format(max_proj_dist_Ab_m))
        print("Minimal distance P_T(Ab)={}\n".format(min_proj_dist_Ab_m))

        print("Projection T_[m+m_1]:")

        proj_dist_m1 = [np.linalg.norm(L_2[:,:m+m_1].dot(L_2[:,:m+m_1].T).dot(A_big[:,i]) - A_big[:,i]) for i in range(m)]
        max_proj_dist_A_m1 = np.max(proj_dist_m1)
        min_proj_dist_A_m1 = np.min(proj_dist_m1)
        results.extend([max_proj_dist_A_m1,min_proj_dist_A_m1])
        print("Maximal distance P_T(A)={}".format(max_proj_dist_A_m1))        
        print("Minimal distance P_T(A)={}".format(min_proj_dist_A_m1))


        projection_dist_Ab_m1 = [np.linalg.norm(L_2[:,:m+m_1].dot(L_2[:,:m+m_1].T).dot(Ab_big[:,i]) - Ab_big[:,i]) for i in range(m_1)]
        max_proj_dist_Ab_m1 = np.max(projection_dist_Ab_m)
        min_proj_dist_Ab_m1 = np.min(projection_dist_Ab_m)
        results.extend([max_proj_dist_Ab_m1,min_proj_dist_Ab_m1])        
        print("Maximal distance P_T(Ab)={}".format(max_proj_dist_Ab_m))
        print("Minimal distance P_T(Ab)={}\n".format(min_proj_dist_Ab_m))



        print("Reconstruction T_[m]")
        coeff = np.zeros(d**2)
        coeff[:m] = D_2[:m]
        v = np.reshape(L_2.dot(coeff), (d,d))
        v = v/np.linalg.norm(v)
        some = algo1(v, L_2[:,:m], gamma=1.5, n = 100)
        lal = np.reshape(some,d**2)/np.linalg.norm(np.reshape(some,d**2))
        print("Distances A")
        dist_A = [np.min([np.linalg.norm(lal-a), np.linalg.norm(lal +a)]) for a in A_tensor]
        print("Minimal: ",np.min(dist_A))
        results.append(np.min(dist_A))
        print("Distances Ab")
        dist_Ab = [np.min([np.linalg.norm(lal-a), np.linalg.norm(lal +a)]) for a in Ab_tensor]
        print("Minimal: ",np.min(dist_Ab))
        results.append(np.min(dist_Ab))


        print("Reconstruction T_[m+m_1]")
        coeff = np.zeros(d**2)
        coeff[:m+m_1] = D_2[:m+m_1]
        v = np.reshape(L_2.dot(coeff), (d,d))
        v = v/np.linalg.norm(v)
        some = algo1(v, L_2[:,:m+m_1], gamma=1.5, n = 100)
        lal = np.reshape(some,d**2)/np.linalg.norm(np.reshape(some,d**2))
        print("Distances A")
        dist_A = [np.min([np.linalg.norm(lal-a), np.linalg.norm(lal +a)]) for a in A_tensor]
        results.append(np.min(dist_A))
        print("Minimal: ",np.min(dist_A))
        print("Distances Ab")
        dist_Ab = [np.min([np.linalg.norm(lal-a), np.linalg.norm(lal +a)]) for a in Ab_tensor]
        print("Minimal: ",np.min(dist_Ab))
        results.append(np.min(dist_Ab))

        return results


if __name__ == "__main__":
    main()