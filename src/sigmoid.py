import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
plotpath = os.path.join(dir_path, "plots")

desc = {'g':["sigmoid"],'dg':["bka"],'ddg':["dtrr"], 'dddg':["test"]}

def plot_sigmoid(filename = "sigmoid.pdf"):
    filepath = os.path.join(plotpath, filename)
    lin = np.linspace(-2,2, 100)
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans')
    _, pltarr = plt.subplots(2,2, figsize=(12,12))
    pltarr[0,0].plot(lin,g(lin))
    pltarr[0,0].set_xlabel(r"$x$")
    pltarr[0,0].axhline(y=0, color="black")
    pltarr[0,0].set_title(r"$\sigma$", fontsize=16)
    pltarr[0,0].set_ylabel(r"$\sigma(x)$")
    pltarr[0,1].plot(lin,dg(lin))
    pltarr[0,1].set_title(r"$\sigma'$", fontsize=16)
    pltarr[0,1].axhline(y=0, color="black")
    pltarr[0,1].set_xlabel(r"$x$")
    pltarr[0,1].set_ylabel(r"$\sigma'(x)$")
    pltarr[1,0].plot(lin,ddg(lin))
    pltarr[1,0].axhline(y=0, color="black")    
    pltarr[1,0].set_title(r"$\sigma''$", fontsize=16)
    pltarr[1,0].set_xlabel(r"$x$")
    pltarr[1,0].set_ylabel(r"$\sigma''(x)$")
    pltarr[1,1].plot(lin,dddg(lin))
    pltarr[1,1].axhline(y=0, color="black")
    pltarr[1,1].set_title(r"$\sigma'''$", fontsize=16)
    pltarr[1,1].set_xlabel(r"$x$")
    pltarr[1,1].set_ylabel(r"$\sigma'''(x)$")
    plt.savefig(filepath, format='pdf', dpi=1200)


def g(x):
    return 1/(1+np.exp(-x))

def dg(x):
    #return sig(x)*(1-sig(x))
    return np.exp(x)/(1+np.exp(x))**2

def ddg(x):
    return -np.exp(x)*(np.exp(x) - 1)/(1+np.exp(x))**3

def dddg(x):
    return np.exp(x)*(1-4*np.exp(x)+np.exp(2*x))/(1+np.exp(x))**4


