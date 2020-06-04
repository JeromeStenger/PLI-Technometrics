import copy
from utils.marshall_olkin_extended_uniform import *
from utils.fisher_information import *
from utils.fisher_triangular import *

# This is the solver used to compute Fisher sphere for parametric models with 2 parameters only (e.g. the Gaussian laws)


def fi_inv(distribution, theta, type="bounded", ngrid=100):
    """
    Helper function to compute an approximation of the inverse Fisher information matrix (FIM) of f in theta,
    based on Simpson's method

    :param distribution: Openturns distribution object
    :param theta: tuple of parameters where to evaluate the Inverse Fisher Information matrix,
                  (if bounded, bounds value needs to be added after the parameters)
    :param type: string, can be "bounded", "left truncated" or "unbounded", This is needed for defining the interval of
                 Simpson's quadrature method
    :return: 2x2 numpy array, The inverse FIM
    """

    f = copy.deepcopy(distribution)

    if type == "bounded":
        a, b = theta[-2], theta[-1]
        q1, q2 = theta[0], theta[1]
        grid = np.linspace(a, b, ngrid)
        f.setParameter((q1, q2, a, b))
    if type == "left truncated":
        a = theta[-1]
        q1, q2 = theta[0], theta[1]
        f.setParameter((q1, q2, a))
        grid = np.linspace(a, a + 10 * f.getStandardDeviation()[0], ngrid)
        f.setParameter((q1, q2, a))

    if type == "unbounded":
        q1, q2 = theta[0], theta[1]
        f.setParameter((q1, q2))
        grid = np.linspace(f.getMean()[0] - 6 * f.getStandardDeviation()[0], f.getMean()[0] + 6 * f.getStandardDeviation()[0], ngrid)
        f.setParameter((q1, q2))

    val = []
    for i in range(2):
        for j in range(2):
            val += [
                [f.computeLogPDFGradient([t])[i] * f.computeLogPDFGradient([t])[j] * f.computePDF(t) for t in grid]]

    for k in range(len(val)):
        val[k] = spi.simps(val[k], grid)
    fi = np.array(val).reshape(2, 2)
    return la.inv(fi)


def sqr(inv_f):
    """
    Helper function to compute the square root of a symmetric definite positive matrix.

    :param inv_f: the matrix to compute the square root
    :return:  2x2 numpy array
    """
    w, v = la.eig(inv_f)
    sqr_inv_f = np.dot(np.dot(v, np.diag(np.sqrt(w))), np.transpose(v))
    return sqr_inv_f


def grad_fi_inv(inv_f, distribution, theta, type="bounded", ngrid=100):
    """
    Helper function to compute the partial derivative of the Inverse FIM wrt to each parameter (with the matrix square root
    method).

    :param inv_f: inverse FIM in theta
    :param f: Openturns distribution object
    :param theta: tuple of parameters where to evaluate the gradient (if bounded, bounds value needs to be added after the parameters)
    :return: a tuple of 2x2 numpy array
    """
    f = copy.deepcopy(distribution)

    s = sqr(inv_f)
    h1 = np.dot(s, np.array([10 ** (-3), 0]).reshape(2, 1))
    h2 = np.dot(s, np.array([0, 10 ** (-3)]).reshape(2, 1))
    h = np.array([h1, h2]).reshape(2, 2)
    if type == "bounded":
        a, b = theta[-2], theta[-1]
        q1, q2 = theta[0], theta[1]
        f2 = fi_inv(f, (q1 + h1[0][0], q2 + h1[1][0], a, b), ngrid=ngrid) - inv_f
        f3 = fi_inv(f, (q1 + h2[0][0], q2 + h2[1][0], a, b), ngrid=ngrid) - inv_f
    if type == "left truncated":
        a = theta[-1]
        q1, q2 = theta[0], theta[1]
        f2 = fi_inv(f, (q1 + h1[0][0], q2 + h1[1][0], a), type="left truncated", ngrid=ngrid) - inv_f
        f3 = fi_inv(f, (q1 + h2[0][0], q2 + h2[1][0], a), type="left truncated", ngrid=ngrid) - inv_f
    if type == "unbounded":
        q1, q2 = theta[0], theta[1]
        f2 = fi_inv(f, (q1 + h1[0][0], q2 + h1[1][0]), type="unbounded", ngrid=ngrid) - inv_f
        f3 = fi_inv(f, (q1 + h2[0][0], q2 + h2[1][0]), type="unbounded", ngrid=ngrid) - inv_f

    gfi0 = np.zeros((2, 2))
    gfi1 = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            g = np.dot(la.inv(h), np.array([f2[i, j], f3[i, j]]).reshape(2, 1))
            gfi0[i, j] = g[0]
            gfi1[i, j] = g[1]
    return gfi0, gfi1


def grad_fi_inv2(inv_f, distribution, theta):
    """
    Helper function to compute the partial derivative of the Inverse FIM wrt to each parameter. (with the optimisation based method)

    :param inv_f: inverse FIM in theta
    :param distribution: Openturns distribution object
    :param theta: Parameter where to evaluate the gradient
    :return: a tuple of 2x2 numpy array
    """

    f = copy.deepcopy(distribution)
    a, b = theta[-2], theta[-1]
    q1, q2 = theta[0], theta[1]
    eps = 10**(-5)
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    h1 = np.dot(inv_f, e1.reshape(2, 1))
    h1 = h1 / np.sqrt(np.dot(e1.T, np.dot(inv_f, e1)))
    h2 = np.dot(inv_f, e2.reshape(2, 1))
    h2 = h2 / np.sqrt(np.dot(e2.T, np.dot(inv_f, e2)))
    h = np.array([h1, h2]).reshape(2, 2)
    f2 = fi_inv(f, (q1 + eps * h1[0][0], q2 + eps * h1[1][0], a, b)) - inv_f
    f3 = fi_inv(f, (q1 + eps * h2[0][0], q2 + eps * h2[1][0], a, b)) - inv_f
    gfi0 = np.zeros((2, 2))
    gfi1 = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            g = np.dot(la.inv(h), np.array([f2[i, j], f3[i, j]]).reshape(2, 1))
            gfi0[i, j] = g[0]
            gfi1[i, j] = g[1]
    return gfi0, gfi1


def hamiltonian(y, t, distribution, theta, type="bounded", ngrid=100):
    """
    Function to pass as argument to scipy.integrate.odeint to compute the geodesic

    :param y: (q1, q2, p1, p2) coordinate of the mobile in the manifold, q1 and q2 match the paramaters,
                p1 and p2 to the "speed" of the particule
    :param t: time argument, mandatory for odeint
    :param f: Openturns distribution object
    :param theta: coordinate + the max and min of the distribution support if type == "bounded"
    :return: the value of the time derivative of y
    """
    f = copy.deepcopy(distribution)
    params = theta[0], theta[1]
    q1, q2, p1, p2 = y
    if type == "bounded":
        inv_f = fi_inv(f, (q1, q2, theta[-2], theta[-1]), type=type, ngrid=ngrid)
        # inv_f = la.inv(fisher_information(f, (q1, q2, 0, theta[2], theta[3]), 2))
        gfi = grad_fi_inv(inv_f, f, (q1, q2, theta[-2], theta[-1]), ngrid=ngrid)
        f.setParameter(theta)
    if type == "left truncated":
        inv_f = fi_inv(f, (q1, q2, theta[-1]), type=type, ngrid=ngrid)
        gfi = grad_fi_inv(inv_f, f, (q1, q2, theta[-1]), type=type, ngrid=ngrid)
    if type == "unbounded":
        inv_f = fi_inv(f, (q1, q2), type=type, ngrid=ngrid)
        gfi = grad_fi_inv(inv_f, f, (q1, q2), type=type, ngrid=ngrid)

    dq = np.dot(inv_f, np.array([p1, p2]))
    dp = [-0.5 * np.dot(np.array([p1, p2]).T, np.dot(gfi[0], np.array([p1, p2]))),
          -0.5 * np.dot(np.array([p1, p2]).T, np.dot(gfi[1], np.array([p1, p2])))]
    return np.array([dq[0], dq[1], dp[0], dp[1]])


def hamiltonian_val(y, distribution, theta, type="unbounded"):
    """
    Compute value of the hamiltonian in y
    :param y: parameter
    :param theta: tuple of parameters
    :param f: Openturns ditribution object
    :return: float value, Hamiltonian value
    """
    f = copy.deepcopy(distribution)
    if type == "unbounded":
        q1, q2, p1, p2 = y
        f.setParameter((q1, q2))
        inv_f = fi_inv(f, (q1, q2), type="unbounded")
        f.setParameter((theta[0], theta[1]))
        return 0.5 * np.dot(np.array([p1, p2]).T, np.dot(inv_f, np.array([p1, p2])))
    if type == "bounded":
        q1, q2, p1, p2 = y
        a, b = theta[2], theta[3]
        f.setParameter((q1, q2, a, b))
        inv_f = fi_inv(f, (q1, q2, a, b), type="bounded")
        f.setParameter(theta)
        return 0.5 * np.dot(np.array([p1, p2]).T, np.dot(inv_f, np.array([p1, p2])))
    if type == "normal":
        q1, q2, p1, p2 = y
        inv_f = np.array([[q2**2, 0], [0, q2**2/2]]).reshape(2, 2)
        return 0.5 * np.dot(np.array([p1, p2]).T, np.dot(inv_f, np.array([p1, p2])))


def hamiltonian_moeu(y, t, theta):
    """
    Compute value of the hamiltonian in y for the Marshall Olkin Extended Uniform family
    :param y: parameter
    :param theta: tuple of parameters
    :param f: Openturns ditribution object
    :return: float value, Hamiltonian value
    """
    q, p = y
    inv_f = fisher_moeu_inv(q, theta[1], theta[2])
    gfi = (fisher_moeu_inv(q + 10 ** (-3) * np.sqrt(inv_f), theta[1], theta[2]) - inv_f) / (10 ** (-3) * np.sqrt(inv_f))
    dq = inv_f * p
    dp = -0.5 * p ** 2 * gfi
    return np.array([dq, dp])


def hamiltonian_triangular(y, t, theta):
    """
    Compute value of the hamiltonian in y for the Triangular family
    :param y: parameter
    :param theta: tuple of parameters
    :param f: Openturns ditribution object
    :return: float value, Hamiltonian value
    """
    q, p = y
    inv_f = fisher_triangular_inv(q, theta[1], theta[2])
    gfi = (fisher_triangular_inv(q + 10 ** (-3) * np.sqrt(inv_f), theta[1], theta[2]) - inv_f) / (10 ** (-3) * np.sqrt(inv_f))
    dq = inv_f * p
    dp = -0.5 * p ** 2 * gfi
    return np.array([dq, dp])


def explicit_euler(fun, y0, t):
    """
    Explicit Euler implementation
    :param fun: function such that y' = fun(y, t)
    :param y0: initial condition
    :param t: time grid
    :return: array of the solution at each time step
    """
    h = float((t[-1] - t[0])/len(t))
    sol = [y0 + h*fun(t[0], y0)]

    for i in range(len(t) - 1):
        sol += [sol[-1] + h*fun(t[i], sol[-1])]

    return np.array(sol)


def symplectic_scheme(y0, t, distribution, omega, type="bounded", bounds=(0,0)):

    """
    Symplectic scheme for non separable hamiltonian (inspired from https://arxiv.org/pdf/1609.02212.pdf)
    :param y0: initial condition
    :param t: time range array in [0, 1]
    :param f: Openturns distribution object
    :param theta: tuple of initial parameters
    :param omega: bounding factor
    :param type: "bounded", "unbounded" or "left truncated"
    :return:
    """
    f = copy.deepcopy(distribution)
    h = float((t[-1] - t[0]) / len(t))
    q1, q2, p1, p2, x1, x2, y1, y2 = y0
    q1, q2 = y0[0], y0[1]
    x1, x2 = y0[4], y0[5]
    if type == "unbounded":
        q = np.array([q1, q2])
        p = np.array([p1, p2])
        x = np.array([x1, x2])
        y = np.array([y1, y2])

        # step 1: time flow of H_A
        f.setParameter((q1, q2))
        inv_f = fi_inv(f, (q1, q2), type=type)
        gfi = grad_fi_inv(inv_f, f, (q1, q2), type=type)

        dp = np.array([0.5 * np.dot(y.T, np.dot(gfi[0], y)), 0.5 * np.dot(y.T, np.dot(gfi[1], y))])
        p = p - h / 2 * dp

        dx = np.dot(inv_f, y)
        x = x + h / 2 * dx

        # step 2: time flow of H_B

        f.setParameter((x1, x2))
        inv_f = fi_inv(f, (x1, x2), type=type)
        gfi = grad_fi_inv(inv_f, f, (x1, x2), type=type)

        dq = np.dot(inv_f, p)
        q = q + h / 2 * dq

        dy = np.array([0.5 * np.dot(p.T, np.dot(gfi[0], p)), 0.5 * np.dot(p.T, np.dot(gfi[1], p))])
        y = y - h / 2 * dy

        # step 3: time flow of omega * H_C

        A = np.cos(2 * omega * h) * np.eye(2)
        B = np.sin(2 * omega * h) * np.eye(2)

        R = np.block([
            [A, B],
            [-B, A]
        ])

        mid = np.block([(q + x)/2, (p + y)/2]).reshape(4, 1)
        dif = np.block([(q - x)/2, (p - y)/2]).reshape(4, 1)
        new1 = mid + np.dot(R, dif)
        new2 = mid - np.dot(R, dif)

        q1, q2, p1, p2 = new1[:, 0]
        x1, x2, y1, y2 = new2[:, 0]

        q = np.array([q1, q2])
        p = np.array([p1, p2])
        x = np.array([x1, x2])
        y = np.array([y1, y2])

        # step 4: time flow of H_B

        f.setParameter((x1, x2))
        inv_f = fi_inv(f, (x1, x2), type=type)
        gfi = grad_fi_inv(inv_f, f, (x1, x2), type=type)

        dq = np.dot(inv_f, p)
        q = q + h / 2 * dq

        dy = np.array([0.5 * np.dot(p.T, np.dot(gfi[0], p)), 0.5 * np.dot(p.T, np.dot(gfi[1], p))])
        y = y - h / 2 * dy

        # step 5: time flow of H_A
        f.setParameter((q1, q2))
        inv_f = fi_inv(f, (q1, q2), type=type)
        gfi = grad_fi_inv(inv_f, f, (q1, q2), type=type)

        dp = np.array([0.5 * np.dot(y.T, np.dot(gfi[0], y)), 0.5 * np.dot(y.T, np.dot(gfi[1], y))])
        p = p - h / 2 * dp

        dx = np.dot(inv_f, y)
        x = x + h / 2 * dx

        q1, q2 = q
        y1, y2 = y
        x1, x2 = x
        p1, p2 = p

    if type == "bounded":
        a, b = bounds
        q = np.array([q1, q2])
        p = np.array([p1, p2])
        x = np.array([x1, x2])
        y = np.array([y1, y2])

        # step 1: time flow of H_A
        f.setParameter((q1, q2, a, b))
        inv_f = fi_inv(f, (q1, q2, a, b), type=type)
        gfi = grad_fi_inv(inv_f, f, (q1, q2, a, b), type=type)

        dp = np.array([0.5 * np.dot(y.T, np.dot(gfi[0], y)), 0.5 * np.dot(y.T, np.dot(gfi[1], y))])
        p = p - h / 2 * dp

        dx = np.dot(inv_f, y)
        x = x + h / 2 * dx

        # step 2: time flow of H_B

        f.setParameter((x1, x2, a, b))
        inv_f = fi_inv(f, (x1, x2, a, b), type=type)
        gfi = grad_fi_inv(inv_f, f, (x1, x2, a, b), type=type)

        dq = np.dot(inv_f, p)
        q = q + h / 2 * dq

        dy = np.array([0.5 * np.dot(p.T, np.dot(gfi[0], p)), 0.5 * np.dot(p.T, np.dot(gfi[1], p))])
        y = y - h / 2 * dy

        # step 3: time flow of omega * H_C

        A = np.cos(2 * omega * h) * np.eye(2)
        B = np.sin(2 * omega * h) * np.eye(2)

        R = np.block([
            [A, B],
            [-B, A]
        ])

        mid = np.block([(q + x) / 2, (p + y) / 2]).reshape(4, 1)
        dif = np.block([(q - x) / 2, (p - y) / 2]).reshape(4, 1)
        new1 = mid + np.dot(R, dif)
        new2 = mid - np.dot(R, dif)

        q1, q2, p1, p2 = new1[:, 0]
        x1, x2, y1, y2 = new2[:, 0]

        q = np.array([q1, q2])
        p = np.array([p1, p2])
        x = np.array([x1, x2])
        y = np.array([y1, y2])

        # step 4: time flow of H_B

        f.setParameter((x1, x2, a, b))
        inv_f = fi_inv(f, (x1, x2, a, b), type=type)
        gfi = grad_fi_inv(inv_f, f, (x1, x2, a, b), type=type)

        dq = np.dot(inv_f, p)
        q = q + h / 2 * dq

        dy = np.array([0.5 * np.dot(p.T, np.dot(gfi[0], p)), 0.5 * np.dot(p.T, np.dot(gfi[1], p))])
        y = y - h / 2 * dy

        # step 5: time flow of H_A
        f.setParameter((q1, q2, a, b))
        inv_f = fi_inv(f, (q1, q2, a, b), type=type)
        gfi = grad_fi_inv(inv_f, f, (q1, q2, a, b), type=type)

        dp = np.array([0.5 * np.dot(y.T, np.dot(gfi[0], y)), 0.5 * np.dot(y.T, np.dot(gfi[1], y))])
        p = p - h / 2 * dp

        dx = np.dot(inv_f, y)
        x = x + h / 2 * dx

        q1, q2 = q
        y1, y2 = y
        x1, x2 = x
        p1, p2 = p

    return q1, q2, p1, p2, x1, x2, y1, y2


def symplectic_second_order(y0, t, f, omega, type="bounded", bounds=(0,0)):

    sol = [y0]
    params = f.getParameter()
    for i in range(len(t) - 1):
        f.setParameter(params)
        sol += [symplectic_scheme(sol[-1], t, f, omega, type=type, bounds=bounds)]

    sol = np.array(sol)
    sol.reshape(len(t), 8)
    f.setParameter(params)

    return sol



def initial_speed(inv_f, delta, npoints):
    """
    Helper function to compute the initial speed vectors necessary to throw our particule at distance delta for t = 1

    :param inv_f: the inverse FIM
    :param delta: the Fisher sphere radius
    :param npoints: number of vectors to compute
    :return:
    """
    grid_t = np.linspace(0, 2 * np.pi, npoints)
    w, v = la.eig(inv_f)
    ellipsoid_eigbasis = [
        np.array([np.cos(t) / np.sqrt(w[0]) * delta, np.sin(t) / np.sqrt(w[1]) * delta])
        for t in grid_t]
    ellipsoid = [np.matmul(v, x) for x in ellipsoid_eigbasis]
    return ellipsoid


def compute_fisher_sphere(args):
    """
    Helper for computing multiple geodesics in parallel
    :param args: (f, y0, theta)
    :return: the coordinate of the Fisher sphere
    """

    f, y0, theta, type = args
    t = np.linspace(0, 1, 100)
    sol = explicit_euler(lambda t, y: hamiltonian(y, t, f, theta, type=type), y0, t)
    return [sol[:, 0][-1], sol[:, 1][-1]]



def compute_fisher_sphere_moeu(args):
    """
    Helper for computing multiple geodesics in parallel
    :param args: (f, y0, theta)
    :return: the coordinate of the Fisher sphere
    """
    f, y0, theta, type = args
    t = np.linspace(0, 1, 100)
    try:
        sol = explicit_euler(lambda t, y: hamiltonian_moeu(y, t, f, theta), y0, t)
        if type == "end":
            return [sol[:, 0][-1]]
        if type == "all":
            return [sol[:, 0]]
    except:
        return [0, 0]


def compute_fisher_sphere_triangular(args):
    """
    Helper for computing multiple geodesics in parallel
    :param args: (f, y0, theta)
    :return: the coordinate of the Fisher sphere
    """
    f, y0, theta = args
    t = np.linspace(0, 1, 100)
    sol = explicit_euler(lambda t, y: hamiltonian_triangular(y, t, f, theta), y0, t)
    return [sol[:, 0][-1]]


