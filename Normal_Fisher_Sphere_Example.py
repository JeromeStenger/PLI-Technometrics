# plotting libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from mpmath import *

from utils.geodesic_solver import *  # Helper functions for computing geodesics
import openturns as ot  # Probabilistic modeling library
    
# %%

# Distribution definition
class PersoNormal(ot.Normal):

    def setParameter(self, parameter):
        super().setParameter([parameter[0], parameter[1]])


# Normal parameters
mu = 0
sigma = 1
f = PersoNormal()
f.setParameter([mu, sigma])

# Truncation Bounds
a = -4
b = 4

# Solver parameters
npoints = 20 # number of distributions sampled on the Fisher sphere
grid_time = 100  # number of time steps
ngrid = 100  # step for computing gradient of the fisher information

delta = [0.5, 1]  # several radius of Fisher spheres
paramSol = np.zeros((len(delta), npoints, 2))

# Figure parameters
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
fig = plt.figure(figsize=(4.5, 3.5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
ax = fig.gca()
ax.set_ylabel(r'$\sigma$', fontsize=10, rotation=0, labelpad=15)
ax.yaxis.set_label_position('left')
ax.set_xlabel(r'$\mu$', fontsize=10)
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# compute all the initials conditions in speed for ODE solving
for k in range(len(delta)):
    p = initial_speed(fi_inv(f, (mu, sigma, a, b), type="bounded", ngrid=ngrid), delta[k], npoints) 
    p1 = [e[0] for e in p]
    p2 = [e[1] for e in p]
    for i in range(npoints):
        y0 = [mu, sigma, p1[i], p2[i]]  # initial condition
        t = np.linspace(0, 1, grid_time)  # time discretization array
        # full geodesic trajectory, the last element of this array belongs to the Fisher sphere of radius delta, 
        # each element is a tuple (mu, sigma)
        try:
            sol1 = explicit_euler(lambda t, y: hamiltonian(y, t, f, (mu, sigma, a, b), type="bounded", ngrid=ngrid), y0, t)
            if sol1[grid_time-1, 1] >= 0:
                paramSol[k, i, :] = np.array([sol1[grid_time-1, 0], sol1[grid_time-1, 1]])
            else:
                paramSol[k, i, :] = [sol1[grid_time-1, 0], 0]
            ax.plot(sol1[:, 0], sol1[:, 1], ls='--')  # plot the geodesic computed
        except:
            paramSol[k, i, :] = [-1, -1]

# Plot the contour of each sphere
color = ['olive', 'steelblue', 'indianred', 'green', 'k', 'dimgray']
for k in range(len(delta)):
    try:
        y = [list(paramSol[k][i]) for i in range(len(paramSol[k]))]
        z = list(filter(lambda a: a != [-1, -1], y))
    except:
        print('')
    ax.plot(np.transpose(z)[0], np.transpose(z)[1], label='delta: '+str(delta[k]), color=color[k])
plt.legend()   
plt.tight_layout()

# %%
# =============================================================================
# ============================= PLOT DENSITIES ================================
# =============================================================================

plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
fig = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
ax = fig.gca()
ax.set_ylabel(r'', fontsize=10, rotation=90, labelpad=20)
ax.yaxis.set_label_position('left')
ax.set_xlabel(r'', fontsize=10)
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

x = np.linspace(a, b, 1000)
k = 0  # index for delta
for i in range(npoints):
    if paramSol[k, i, 1] > 0:    
        f = ot.Normal(paramSol[k, i, 0], paramSol[k, i, 1])
        y = [f.computePDF(x[j]) for j in range(len(x))]
        plt.plot(x, y, color='b', alpha=0.5)
plt.plot(x, [ot.Normal(mu, sigma).computePDF(x[j]) for j in range(len(x))], color='r', linestyle='dashed', linewidth=2)

plt.tight_layout()   