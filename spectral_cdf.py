import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def scdf(hessian, plotname="Test_CDF", logx=False, debug=False):
    """
    Compute the spectral cdf of a matrix
    Mean to be used alongside a sketched Hessian
    """
    eigenvalues = np.linalg.eigvalsh(hessian)
    d = len(eigenvalues)
    xpoints = []
    ypoints = []
    cdf_val = 1
    for l in eigenvalues:
    	ypoints.append(cdf_val)
    	xpoints.append(l)
    	cdf_val -= 1/d
    	ypoints.append(cdf_val)
    	xpoints.append(l)
    plt.plot(xpoints, ypoints)
    if logx:
    	plt.xscale('symlog')
    plt.ylabel('Cumulative Spectral Density')
    plt.xlabel('Eigenvalue')
    plt.axis([np.min(eigenvalues), np.max(eigenvalues), 0, 1])
    plt.tight_layout()
    plt.savefig(plotname)

def density_to_scdf(eigen_density, eigen_values, plotname="Test_CDF", logx=False, debug=False):
    """
    Compute the spectral cdf based on the 
    Mean to be used as a follow-up from the original density function
    """
    eigen_density = eigen_density/sum(eigen_density)
    xpoints = []
    ypoints = []
    cdf_val = 1
    for l_i in range(len(eigen_values)):
    	l = eigen_values[l_i]
    	ypoints.append(cdf_val)
    	xpoints.append(l)
    	cdf_val -= eigen_density[l_i]
    	ypoints.append(cdf_val)
    	xpoints.append(l)
    plt.plot(xpoints, ypoints)
    if logx:
    	plt.xscale('symlog')
    plt.ylabel('Cumulative Spectral Density')
    plt.xlabel('Eigenvalue')
    plt.axis([eigen_values[0], eigen_values[-1], 0, 1])
    plt.tight_layout()
    plt.savefig(plotname)
    return ""