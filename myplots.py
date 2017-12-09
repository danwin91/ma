import matplotlib.pyplot as plt 
import os
import numpy as np

#Saving figures to
dir_path = os.path.dirname(os.path.realpath(__file__))
plotpath = os.path.join(dir_path, "plots")
print("[myplots.py]: Saving plots to {}".format(plotpath))



def plot_singular_values(D, m, m_1, m_x, der, filename = None):
    plt.plot(list(range(1,1+len(D))), D, "bx-", label=r'singular values')
    plt.axvline(x=m+m_1, color="red", linestyle="--", label=r'$m+m_1$')
    plt.axvline(x=2*m, color="green", linestyle="--", label=r"$2m$")
    plt.title(s=r'Singular values of $M_{}$ for $m={},m_1={}, m_x={}$'.format(der,m,m_1,m_x),  fontsize=16)
    plt.legend(loc='upper right')
    if filename:
        filepath = os.path.join(plotpath, filename)
        plt.savefig(filepath, format='pdf', dpi=1200)

def plot_summary_dist(m_vec, y1, y1_label, y2, y2_label, y3, y3_label,title,  filename = None):
    plt.plot(m_vec, y1, "bx-", label=y1_label)
    plt.plot(m_vec, y2, "rx-", label=y2_label)
    plt.plot(m_vec, y3, "gx-", label=y3_label)
    plt.xlabel(r"$m$")
    plt.title(s=title,  fontsize=16)
    plt.legend(loc='upper right')
    if filename:
        filepath = os.path.join(plotpath, filename)
        plt.savefig(filepath, format='pdf', dpi=1200)


def plot_summary_ratio(m_vec, y1, y1_label, y2, y2_label, title, filename = None):
    plt.plot(m_vec, y1, "bx-", label=y1_label)
    plt.plot(m_vec, y2, "rx-", label=y2_label)
    plt.xlabel(r"$m$")
    plt.ylabel(r"$ratio(D_k, j)$")
    plt.title(s=title,  fontsize=16)
    plt.legend(loc='upper right')
    if filename:
        filepath = os.path.join(plotpath, filename)
        plt.savefig(filepath, format='pdf', dpi=1200)

def plot_ratio(D,m,m_1, filename = None):
    D_vec = [np.sum(D[:k])/np.sum(D) for k in range(1, 2*m+m_1)]
    plt.plot(range(1,2*m+m_1), np.log(D_vec), "bx-", label="$ratio$")
    plt.xlabel(r"$j$")
    plt.axvline(x=m+m_1, color="red", linestyle="--", label=r'$m+m_1$')
    plt.axvline(x=2*m+m_1, color="red", linestyle="-", label=r'$2m+m_1$')
    plt.ylabel(r"$ratio(D_k, j)$")
    plt.title(s="fuckyeah",  fontsize=16)
    plt.legend(loc='lower right')
    if filename:
        filepath = os.path.join(plotpath, filename)
        plt.savefig(filepath, format='pdf', dpi=1200)