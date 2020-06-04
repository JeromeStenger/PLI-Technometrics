import openturns as ot
import numpy as np
import scipy.integrate as spi


def fisher_triangular_inv(c, a, b):
    grid = np.linspace(a, b, 100)
    f = ot.Triangular(a, c, b)
    val = [f.computeLogPDFGradient([t])[1] ** 2 * f.computePDF([t]) for t in grid]
    fi = spi.simps(val, grid)
    return 1 / fi

