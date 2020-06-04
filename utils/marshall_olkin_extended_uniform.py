import numpy as np
import scipy.integrate as spi


def moeu(alpha, a, b, x):
    return alpha * (b - a) / (alpha * b - a + (1 - alpha) * x) ** 2


def fisher_moeu_inv(alpha, a, b):
    sample = np.linspace(a, b, 100)
    val = np.array(
        [((np.log(moeu(alpha + 0.01, a, b, x)) - np.log(moeu(alpha, a, b, x)))/0.01) ** 2 * moeu(alpha, a, b, x) for x in
         sample])
    fi = spi.simps(val, sample)
    return 1/fi


class Moeu:

    def __init__(self, alpha, a, b):
        self.alpha = alpha
        self.a = a
        self.b = b

    def computePDF(self, x):
        return moeu(self.alpha, self.a, self.b, x)
