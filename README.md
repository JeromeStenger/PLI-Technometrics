# Geodesics computation with Fisher metric

This module provides geodesics computation with respect to the Fisher Information in a parametric model.


## Tutorial

The code below computes several geodesics for the Gaussian parametric model.

```python

# plotting libraries
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from utils.geodesic_solver import *  # Helper functions for computing geodesics
import openturns as ot  # Probabilistic modeling library

# example with the Gaussian geometry

delta = 1  # radius of the Fisher sphere
sigma = 1
f = ot.Normal(0, sigma)
npoints = 25  # number of distributions sampled on the Fisher sphere
grid_time = 100  # number of time steps

# compute all the initials conditions in speed for ODE solving
p = initial_speed(fi_inv(f, (0, sigma), type="unbounded"), delta, npoints) 

p1 = [e[0] for e in p]
p2 = [e[1] for e in p]

for i in range(npoints):
    y0 = [0, sigma, p1[i], p2[i]]  # initial condition
    t = np.linspace(0, 1, grid_time)  # time discretization array
    
    # full geodesic trajectory, the last element of this array belongs to the Fisher sphere of radius delta, 
    # each element is a tuple (mu, sigma)
    sol = explicit_euler(lambda t, y: hamiltonian(y, t, f, (0, sigma), type="unbounded"), y0, t) 
    
    plt.plot(sol[:, 0], sol[:, 1])  # plot the geodesic computed
```