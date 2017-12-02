import matplotlib.pyplot as plt 
import os

#Saving figures to
dir_path = os.path.dirname(os.path.realpath(__file__))
plotpath = os.path.join(dir_path, "plots")
print("[myplots.py]: Saving plots to {}".format(plotpath))



def plot_singular_values(D, m, m_1, m_x, der, filename):
    filepath = os.path.join(plotpath, filename)
    plt.plot(list(range(1,1+len(D))), D, "bx-", label=r'singular values')
    plt.axvline(x=m+m_1, color="red", linestyle="--", label=r'$m+m_1$')
    plt.axvline(x=2*m, color="green", linestyle="--", label=r"$2m$")
    plt.title(s=r'Singular values of $M_{}$ for $m={},m_1={}, m_x={}$'.format(der,m,m_1,m_x),  fontsize=16)
    plt.legend(loc='upper right')
    plt.savefig(filepath, format='pdf', dpi=1200)