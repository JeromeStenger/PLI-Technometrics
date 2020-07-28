from utils.geodesic_solver import *
from utils.pli import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openturns as ot


def ishigami(x):
    return np.sin(x[0]) + 7 * np.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * np.sin(x[0])


# Definition of the input distribution
mu = 0
sigma = 1
distribution = [ot.Normal(mu, sigma)]*3

# Sampling in the input distribution
N = 2000
delta = [1]
alpha = 0.95
sample = np.array([d.getSample(N) for d in distribution])[:, :, 0]
g = [ishigami(sample[:, i]) for i in range(N)]

# Computation of the initial quantile
grid = np.linspace(np.min(g), np.max(g), 500)
q = quantile(0.95, empirical_cdf(g, grid), grid)

# Parameters for computing the geodesics
npoints = 10  # number of distributions sampled on the Fisher sphere
grid_time = 100  # number of time steps

index = [0, 1, 2]  # index numbering the input, vary in {0;1;2}
delta = np.linspace(0, 1, 10)  # range of perturbation

p_max = np.zeros([len(index), len(delta)])
p_max_sup = np.zeros([len(index), len(delta)])
p_max_inf = np.zeros([len(index), len(delta)])


p_min = np.zeros([len(index), len(delta)])
p_min_sup = np.zeros([len(index), len(delta)])
p_min_inf = np.zeros([len(index), len(delta)])
boot = np.random.randint(0, N - 1, size=(100, 200))
for k in range(len(delta)):
    print(f"delta = {delta[k]}")
    p_maxb = []
    p_minb = []
    for j in index:
        p_hat = []
        p = initial_speed(fi_inv(distribution[j], (mu, sigma), type="unbounded"), delta[k], npoints) 
        p1 = [e[0] for e in p]
        p2 = [e[1] for e in p]
        paramSol = [[]]*npoints
        for i in range(npoints):
            y0 = [mu, sigma, p1[i], p2[i]]  # initial condition
            t = np.linspace(0, 1, grid_time)  # time discretization array
            # full geodesic trajectory, the last element of this array belongs to the Fisher sphere of radius delta,
            # each element is a tuple (mu, sigma)
            sol = explicit_euler(lambda t, y: hamiltonian(y, t, distribution[j], (mu, sigma), type="unbounded"), y0, t)
            paramSol[i] = [sol[grid_time-1, 0], sol[grid_time-1, 1]]
            # plt.plot(sol[:, 0], sol[:, 1])  # plot the geodesic computed
            q_delta = quantile(alpha, ris_cdf_estimator(sample[j, :], grid, distribution[j], ot.Normal(paramSol[i][0], paramSol[i][1]), g, "ot"), grid)
            p_hat += [pli(q, q_delta)]
        min_idx = np.argmin(p_hat)
        max_idx = np.argmax(p_hat)

        for b in range(200):
            # print(f"b = {b}")

            q_delta_min = quantile(alpha,
                ris_cdf_estimator([sample[j, i] for i in boot[:, b]], grid, distribution[j], ot.Normal(paramSol[min_idx][0], paramSol[min_idx][1]), [g[i] for i in boot[:, b]], "ot"), grid)
            p_minb += [pli(q, q_delta_min)]

            q_delta_max = quantile(alpha,
                ris_cdf_estimator([sample[j, i] for i in boot[:, b]], grid, distribution[j],
                                  ot.Normal(paramSol[max_idx][0], paramSol[max_idx][1]), [g[i] for i in boot[:, b]], "ot"), grid)
            p_maxb += [pli(q, q_delta_max)]

        p_max[j, k] = np.mean(p_maxb)
        p_max_sup[j, k] = np.quantile(p_maxb, 0.95)
        p_max_inf[j, k] = np.quantile(p_maxb, 0.05)

        p_min[j, k] = np.mean(p_minb)
        p_min_sup[j, k] = np.quantile(p_minb, 0.95)
        p_min_inf[j, k] = np.quantile(p_minb, 0.05)

        name = 'Ishigami_pmin'
        my_df = pd.DataFrame(p_min)
        my_df.to_csv('./' + name + '.csv', index=False, header=False, sep=' ')
        name = 'Ishigami_pmax'
        my_df = pd.DataFrame(p_max)
        my_df.to_csv('./' + name + '.csv', index=False, header=False, sep=' ')

        name = 'Ishigami_pmin_inf'
        my_df = pd.DataFrame(p_min_inf)
        my_df.to_csv('./' + name + '.csv', index=False, header=False, sep=' ')
        name = 'Ishigami_pmax_inf'
        my_df = pd.DataFrame(p_max_inf)
        my_df.to_csv('./' + name + '.csv', index=False, header=False, sep=' ')

        name = 'Ishigami_pmin_sup'
        my_df = pd.DataFrame(p_min_sup)
        my_df.to_csv('./' + name + '.csv', index=False, header=False, sep=' ')
        name = 'Ishigami_pmax_sup'
        my_df = pd.DataFrame(p_max_sup)
        my_df.to_csv('./' + name + '.csv', index=False, header=False, sep=' ')


# %%
# =============================================================================
# ============================ PLOT THE RESULTS ===============================
# =============================================================================

'''
Our results are available in the folder Results/ and have been plotted with
npoints=100 points in the Fisher sphere with radius varying in delta=np.linspace(0.1, 1, 30)
'''
index = [0, 1, 2]
delta = np.linspace(0.1, 1, 30)

# Data importation
p_min = pd.read_csv('./Results/Ishigami_pmin.csv', header=None, sep=' ')
p_min = p_min.values
p_max = pd.read_csv('./Results/Ishigami_pmax.csv', header=None, sep=' ')
p_max = p_max.values

# Figure parameters
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']
fig = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
ax = fig.gca()
ax.set_ylabel(r'PLI', fontsize=10, rotation=90, labelpad=10)
ax.yaxis.set_label_position('left')
ax.set_xlabel(r'$\delta$', fontsize=10)
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

label = [r'$\widehat{S}^+_{N,1\delta} \ \&\ \widehat{S}^-_{N,1\delta} $', r'$\widehat{S}^+_{N,2\delta} \ \&\ \widehat{S}^-_{N,2\delta}$', r'$\widehat{S}^+_{N,3\delta} \ \&\ \widehat{S}^-_{N,3\delta}$']
color = ['r', 'b', 'g']
linestyle = ['-', '--', ':']

# Plot the results
for i in range(len(index)):
    plt.plot(delta, p_min[i], label=label[i], color=color[i], linestyle=linestyle[i])
    plt.plot(delta, p_max[i], color=color[i], linestyle=linestyle[i])

plt.legend()
plt.tight_layout()
