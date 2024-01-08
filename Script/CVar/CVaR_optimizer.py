#######################################################
#### CVar Optimization: Rockafellar et al. (2000) #####
#######################################################

import pandas as pd
from gurobipy import Model, GRB, quicksum


ESG_score = pd.read_csv("Output/DATA/ESG_Score.csv", index_col=0)

def CVar(beta, R, returns_mat, target_vol, mu, Sigma,
         ESG_lb, strict = False, include_target_vol = True,
         include_ESG_rating_target = True, esg_score = ESG_score):
    """
    :param ESG_lb: Constrain for ESG weighted average score
    :param esg_score: List of ESG_Score. Shouldn't be changed.
    :param include_ESG_rating_target: If True, include ESG score constrain.
    :param beta: The confidence level (ex: 90% then 0.9)
    :param R: Target Return (should be consistent with the scale of returns_mat)
    :param returns_mat: Dataframe of returns. In this context it should be simulated. (non-) parametric.
    :param target_vol: Target volatility
    :param mu: Population mean of returns
    :param Sigma: Population covariance of returns
    :param strict: If True, the target vol constrain will be set to == target_vol. O/w by default it will be
                    set to <= target_vol
    :param include_target_vol: If False, no target_vol constrain
    :return: weight and alpha.
             simulate_returns: the optimal portfolio loss (-1 * x * y) given returns_mat
             CVar: the optimized CVar of the portfolio
    """

    # Create a Gurobi model
    model = Model("CVaR_Opt")

    # Necessary Parameters
    q = returns_mat.shape[0]
    n_asset = returns_mat.shape[1]


    # Variables
    u = pd.Series(model.addVars(q))
    x = pd.Series(model.addVars(returns_mat, lb=0, ub=1))
    alpha = pd.Series(model.addVars(1))

    # Objective function
    obj = alpha[0] + 1 / ((1 - beta) * q) * quicksum(u)

    # Add Constrains
    model.addConstrs(x.iloc[i] >= 0 for i in range(x.shape[0]))
    model.addConstr(quicksum(x) == 1)
    model.addConstr((-1) * x.T @ mu <= -R)
    model.addConstrs(u[i] >= 0 for i in range(u.shape[0]))
    model.addConstrs(quicksum([x.T @ list(returns_mat.iloc[i, :]),
                               alpha[0],
                               u[i]]) >= 0 for i in range(q))

    # Optional Constrains
    # (1) Target volatility
    if include_target_vol:
        if strict:
            model.addConstr(x.T @ Sigma @ x == target_vol ** 2)
            model.setParam("NonConvex", 2)  # Strict quadratic constrain  (e.g. ==)  is never convex
        else:
            model.addConstr(x.T @ Sigma @ x <= target_vol ** 2)

    # (2) ESG score constrains
    if include_ESG_rating_target:
        model.addConstr(x.T @ list(esg_score.iloc[:,0]) >= ESG_lb)


    # Optimization
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if model.status == 3:
        return None, None, None, None


    # Output
    w = [x[i].x for i in range(n_asset)] # portfolio weight
    a = alpha[0].x # the by-product optimal VaR TODO: need theoretical justification of optimality
    simulate_returns = [(-1) * (w @ returns_mat.iloc[i, :]) for i in range(q)]
    cvar = model.objVal

    return cvar, w, a, simulate_returns



# TODO: 1. Think of how to automatically set a range of mu s.t. no need to try the upper bond everytime.
# TODO: 2. Add ESG rating constrain (see word doc.)