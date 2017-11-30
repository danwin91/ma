import matplotlib.pyplot as plt 



def plot_singular_values(D, m, m_1, m_x):
    plt.plot(list(range(1,1+len(D))), D, "bx-", label=r'singular values')
    plt.axvline(x=m+m_1, color="red", linestyle="--", label=r'$m+m_1$')
    plt.axvline(x=2*m, color="green", linestyle="--", label=r"$2m$")
    plt.title(s=r'Singular values of $M_1$ for $m={},m_1={}, m_x={}$'.format(m,m_1,m_x),  fontsize=16)
    plt.legend(loc='upper right')