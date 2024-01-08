import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.experimental import enable_iterative_imputer
import yfinance as yf
from sklearn.impute import IterativeImputer
import random
from Script.ETFs.ETF import ETF
import pandas_datareader as pdr
from Script.CVar.CVaR_optimizer import CVar
from Script.CVar.bootstraper import bs_parametric_MVN
from Script.MinVar.MinVar_optimizer import  MinVar


# bs_r, bs_cov = bs_parametric_MVN(r_ETF)

# def CVar(beta, R, returns_mat, target_vol, mu, Sigma,
#          ESG_lb, strict = False, include_target_vol = True,
#          include_ESG_rating_target = True, esg_score = ESG_score):



r_ETF = pd.read_csv("Output/DATA/ETFs_month_returns.csv", index_col=0, parse_dates=True)
r_ETF = r_ETF.iloc[100:,:]
r_ETF,_ = bs_parametric_MVN(r_ETF)

sigma_list = []
m_space = np.linspace(r_ETF.mean()[r_ETF.mean() > 0].min(), r_ETF.mean().max(), 40)
for m in m_space:
    cvar, w, car, _ = CVar(beta = 0.9, R = m, returns_mat = r_ETF, target_vol=0.15/np.sqrt(12),
                           mu = r_ETF.mean(), ESG_lb=6,
                           Sigma=r_ETF.cov(), include_target_vol = False)

    mu_0 = w @ r_ETF.mean()
    sigma = np.sqrt(w @ r_ETF.cov() @ w)
    sigma_list.append(sigma)


sigma_list_0 = []
for m in m_space:
    w = MinVar(m, np.array(r_ETF.mean()),  r_ETF.cov(), ESG_lb=0)

    mu_0 = w.T @ r_ETF.mean()
    sigma = np.sqrt(w.T @ r_ETF.cov() @ w)
    sigma_list_0.append(sigma.values[0][0])

plt.plot(sigma_list, m_space, label = "CVaR")
plt.plot(sigma_list_0, m_space, label = "MV")
plt.title("Efficient Frontier for CVaR and MV")
plt.legend()
plt.gcf()
plt.savefig("Output/FIGURE/CVAR_MV_EF0.png", dpi = 200)
plt.show()





