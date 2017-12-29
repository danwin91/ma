import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
plotpath = os.path.join(dir_path, "plots")

def plot_tanh(filename = "tanh.pdf"):
    filepath = os.path.join(plotpath, filename)
    lin = np.linspace(-2,2, 100)
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans')
    _, pltarr = plt.subplots(2,2,figsize=(12,12))
    pltarr[0,0].plot(lin,g(lin))
    pltarr[0,0].set_xlabel(r"$x$")
    pltarr[0,0].axhline(y=0)
    pltarr[0,0].set_title(r"$\sigma$", fontsize=16)
    pltarr[0,0].set_ylabel(r"$\sigma(x)$")
    pltarr[0,1].plot(lin,dg(lin))
    pltarr[0,1].set_title(r"$\sigma'$", fontsize=16)
    pltarr[0,1].axhline(y=0)
    pltarr[0,1].set_xlabel(r"$x$")
    pltarr[0,1].set_ylabel(r"$\sigma'(x)$")
    pltarr[1,0].plot(lin,ddg(lin))
    pltarr[1,0].axhline(y=0)    
    pltarr[1,0].set_title(r"$\sigma''$", fontsize=16)
    pltarr[1,0].set_xlabel(r"$x$")
    pltarr[1,0].set_ylabel(r"$\sigma''(x)$")
    pltarr[1,1].plot(lin,dddg(lin))
    pltarr[1,1].axhline(y=0)
    pltarr[1,1].set_title(r"$\sigma'''$", fontsize=16)
    pltarr[1,1].set_xlabel(r"$x$")
    pltarr[1,1].set_ylabel(r"$\sigma'''(x)$")
    plt.savefig(filepath, format='pdf', dpi=1200)

def g(x):
    return np.tanh(x)

def dg(x):
    #return sig(x)*(1-sig(x))
    return 1/(np.cosh(x)**2)

def ddg(x):
    return -2*np.sinh(x)/(np.cosh(x)**3)

def dddg(x):
    return 2*(-2 + np.cosh(2*x))/(np.cosh(x)**4)


if __name__ == "__main__":
    print(dg(0))