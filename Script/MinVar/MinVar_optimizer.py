#############################################
#######  Classic MinVar Optimization  ######
#############################################

import cvxpy as cp
import numpy as np
import pandas as pd
from cvxpy import ECOS

ESG_score = pd.read_csv("Output/DATA/ESG_Score.csv", index_col=0)
ESG_score = np.array(ESG_score)

def MinVar(mu_0, z, cov, ESG_lb = 6):

    """
    All input should be np.arrays. O/w the returned object will be of various types
        like pd.dataframe etc.
    mu_0: Target return in R.
    z: np.array. Expected returns, in R^n
    cov: np.array. var and cov matrix
    return: list of [weight, return, variance]
    """

    n = z.shape[0]
    x = cp.Variable((n, 1))
    ones = np.ones([n, 1])

    # Objective function
    cons = [z.T @ x == mu_0,
            ones.T @ x == 1,
            x >= 0,
            x.T @ ESG_score >= ESG_lb
            ]

    problem = cp.Problem(cp.Minimize(cp.quad_form(x, cov)),
                         cons)
    problem.solve(solver = ECOS, verbose=True)

    if problem.status == "infeasible":

        n = z.shape[0]
        x = cp.Variable((n, 1))
        ones = np.ones([n, 1])

        cons = [z.T @ x == mu_0,
                ones.T @ x == 1,
                x >= 0
                ]
        problem = cp.Problem(cp.Minimize(cp.quad_form(x, cov)),
                             cons)
        problem.solve(solver=ECOS, verbose=True)

    return x.value



