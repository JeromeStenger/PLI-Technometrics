import numpy as np
import openturns as ot


def empirical_cdf(sample, grid):

    cdf = np.zeros((len(grid), len(sample)))

    G = np.array([sample, ] * len(grid))
    T = np.array([grid, ] * len(sample))

    B = G <= T.transpose()

    return np.sum(B, axis=1) / len(sample)


def ris_cdf_estimator(sample, grid, f, f_delta, g, dis_type="ot"):
    """
    RIS (Reverse Importance Sampling) estimator of the CDF of g with distribution f_delta

    :param sample: list of size N of values sampled with f
    :param grid: point where to evaluate CDF
    :param f: original distribution
    :param f_delta: perturbated distribution
    :param g: list of size N of the evaluation of the numerical model with sample
    :param dis_type: "ot" if f_delta is a Openturns distribution, "custom" if f_delta custom function
    :return: float value, CDF RIS estimator in t
    """


    if dis_type == "ot":

        cdf = np.zeros((len(grid), len(sample)))
        likelihood_ratio = np.array([f_delta.computePDF(x) / f.computePDF(x) for x in sample])

        G = np.array([g, ] * len(grid))
        T = np.array([grid, ] * len(sample))

        B = G <= T.transpose()

        L = np.array([likelihood_ratio, ] * len(grid))
        B = L * B

        return np.sum(B, axis=1) / np.sum(likelihood_ratio)



    if dis_type == "custom":

        cdf = np.zeros((len(grid), len(sample)))
        likelihood_ratio = np.array([f_delta(x) / f.computePDF(x) for x in sample])

        G = np.array([g, ] * len(grid))
        T = np.array([grid, ] * len(sample))

        B = G <= T.transpose()

        L = np.array([likelihood_ratio, ] * len(grid))
        B = L * B

        return np.sum(B, axis=1) / np.sum(likelihood_ratio)



def rosenblatt_cdf_estimator(sample, t, f, delta, g):
    """
    RIS (Reverse Importance Sampling) estimator of the CDF of g with distribution f_delta

    :param sample: list of size N of values sampled with f
    :param t: point where to evaluate CDF
    :param f: original distribution
    :param f_delta: perturbated distribution
    :param g: list of size N of the evaluation of the numerical model with sample
    :return: float value, CDF RIS estimator in t
    """
    i = 0
    denominator = np.sum([np.exp((2*delta*ot.Normal().computeQuantile(f.computeCDF(x))[0] - delta**2)/2) for x in sample])
    j = 0
    for x in sample:
        if g[j] <= t:
            i += np.exp((2*delta*ot.Normal().computeQuantile(f.computeCDF(x))[0] - delta**2)/2)
        j += 1
    return i / denominator


def quantile(alpha, cdf, grid):
    """
    Compute quantile of level alpha

    :param alpha: level of the quantile
    :param cdf: list of size T of the evaluation of cdf in grid
    :param grid: list of size T
    :return: float, the quantile of level alpha for the empirical CDF cdf
    """

    assert len(grid) == len(cdf)

    boolean = cdf <= alpha
    return grid[np.sum(boolean)]


def pli(q, q_delta):
    """
    Compute the PLI

    :param q: alpha quantile of the original distribution
    :param q_delta: alpha quantile of the perturbated distribution
    :return: float, value of the PLI
    """
    return q_delta / q - 1
    


def compute_pli(args):
    """
    Helper function to compute the PLI with the multiprocessing library

    :param args: 6 - uplet corresponding of the f sample, the original quantile, f, f_delta and the numerical code g
    :return: float, pli value
    """
    sample, q, grid, f, f_delta, g, dis_type = args
    if dis_type == "ot":
        q_delta = quantile(0.95,
                           [ris_cdf_estimator(sample, t, f, f_delta, g) for
                            t in grid],
                           grid)
        return pli(q, q_delta)

    if dis_type == "custom":
        cdf = [ris_cdf_estimator(sample, t, f, f_delta, g, "custom") for
                            t in grid]
        q_delta = quantile(0.95,
                           cdf,
                           grid)
        del cdf
        return pli(q, q_delta)
