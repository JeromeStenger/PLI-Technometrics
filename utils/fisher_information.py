import numpy as np
import multiprocessing
import numpy.linalg as la



def emp_fisher(args):
    """
    Helper function to compute loglikgrad * loglikgrad^T where loglikgrad is the gradient of the score function at point
    theta, evaluate in the point

    :param point: sample point where to evaluate
    :param f:
    :param theta:
    :return:
    """
    point, f, theta, d, dis_type = args
    f.setParameter(theta)
    x = np.array(f.computeLogPDFGradient([point]))
    y = []
    for i in range(d):
        y += [x[0][i]]
    y = np.array(y)
    y = y.reshape(d, 1)
    return np.matmul(y, np.transpose(y))


def fisher_information(f, theta, d, dis_type="ot"):
    """
    Compute the Fisher Information matrix at point theta for the distribution f

    :param f: Statistical distribution (openturns)
    :param theta: a list or numpy array of dimension d = number of parameters
    :param d: number of parameters
    :param dis_type: to precise if f is it a openturns distribution or a custom function

    :return: a (d, d) numpy array
    """

    # Get a large sample distributed by f(theta)

    N = 1000
    f.setParameter(theta)
    sample = f.getSample(N)

    # Compute gradient log likelihood
    pool = multiprocessing.Pool(100)
    args = [(point, f, theta, d, dis_type) for point in sample]
    fisher_info = pool.map(emp_fisher, args)
    fisher_info = np.mean(fisher_info, axis=0)

    # Fisher Information Matrix is always nonnegative, replacing negative values due to statistical fluctuation by
    # zeroes
    return np.array([0 if fisher_info[i, j] < 0 else fisher_info[i, j] for i in range(d) for j in range(d)]).reshape(d,
                                                                                                                     d)


def fisher_ellipsoid(f, theta, delta, npoints, d):
    """
    Compute approximate sphere of radius delta w.r.t. the Fisher information at theta.

    :param f: distribution family
    :param theta: center point
    :param delta: radius of the sphere
    :param npoints: number of points to compute on the sphere
    :param d: number of parameters of the manifold
    :return: list of size npoints x 2
    """

    grid_t = np.linspace(0, 2 * np.pi, npoints)
    fi = fisher_information(f, theta, d)
    w, v = la.eig(fi)
    new_theta = np.matmul(np.transpose(v), np.array(theta[:d]))
    ellipsoid_eigbasis = [
        np.array([new_theta[0] + np.cos(t) / np.sqrt(w[0]) * delta, new_theta[1] + np.sin(t) / np.sqrt(w[1]) * delta])
        for t in grid_t]
    ellipsoid = [np.matmul(v, x) for x in ellipsoid_eigbasis]
    return ellipsoid


