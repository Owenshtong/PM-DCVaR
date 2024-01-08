###########################################
## Function to get the l1-filtered trend ##
###########################################

import cvxpy as cpy
import numpy as np
import cvxpy as cp
import scipy as scipy


def l1_filter(yt, lbd):
    """
    :param yt: Original time series. Should be a list/ array of value
    :param lbd: lambda. Smoothness control. Must be positive
    :return: The filtered time series
    """

    n = len(yt)
    e = np.ones((1, n))
    D = scipy.sparse.spdiags(np.vstack((e, -2 * e, e)), range(3), n - 2, n)

    # Variable
    x = cp.Variable(shape=n)

    # Objective function
    obj = cp.Minimize(0.5 * cp.sum_squares(yt - x)
                      + lbd * cp.norm(D * x, 1))

    prob = cp.Problem(obj)

    # Solve
    prob.solve(solver=cp.CVXOPT, verbose=True)

    return x.value



