############################################################
### FIN 60202 Project 3: A CVaR optimization Approach to ###
###               ESG concerning ETF Allocation          ###
############################################################

# PIP system packages
import pandas as pd
import numpy as np


# Self-defined objects and functions
from Script.DAA import functions as DAA
from Script.Backtest import WalkForward
from Script.MinVar.MinVar_optimizer import MinVar

### Part 0: Load Data
r_ETF = pd.read_csv("Output/DATA/ETFs_month_returns.csv", index_col=0, parse_dates=True)
g = pd.read_csv("Output/DATA/growth_indicator.csv", index_col=0, parse_dates=True)
pi = pd.read_csv("Output/DATA/inflation_indicator.csv", index_col=0, parse_dates=True)

g = g.iloc[:,0]
pi = pi.iloc[:,0]

# Synchronize date of all data: Start from & end in the same date available
t_0 = max(r_ETF.index[0], g.index[0], pi.index[0])
t_T = min(r_ETF.index[-1], g.index[-1], pi.index[-1])

r_ETF = r_ETF[t_0:t_T]
g = g[t_0:t_T]
pi = pi[t_0:t_T]

# Time path
_0_T = list(r_ETF.index)

### Part 1 Back Testing Regime CVaR: 6-Fold Cross Validation WalkForward ###
regime_0T = pd.DataFrame(DAA.classify_1lag_vect(g - g.shift(1),pi - pi.shift(1))).rename(columns =  {0 : "regime"})
regime_0T.index = _0_T
regime_0T = regime_0T.iloc[1:]
regime_0T["Indicator"] = regime_0T.index


# Count at each t, distribution of regimes
regime_count = pd.DataFrame(columns = ["Goldilocks", "Heating_up", "Slow_growth", "Stagflation"])
for i in range(regime_0T.shape[0]):
    temp = regime_0T.iloc[0:i+1].groupby(by="regime").count().T.reset_index(drop = True)
    regime_count = pd.concat([regime_count, temp])
regime_count.index = _0_T[1:]


# Part 1: Backtesting: CV ################# Don't run ##################
points = ["2000-12-01","2005-02-01", "2010-12-01", "2019-12-01"]
r_ETF_train_test = WalkForward.train_test_split(r_ETF, points)
g_train_test =  WalkForward.train_test_split(g, points)
pi_train_test = WalkForward.train_test_split(pi, points)


keys = list(r_ETF_train_test.keys())
opt_cvar_pd_list = []
opt_mv_pd_list = []
for key in keys:
    opt_cvar_pd, opt_mv_pd = WalkForward.cv_once(r_ETF_train_test[key],
                    g_train_test[key],
                    pi_train_test[key], test_start=key)
    opt_cvar_pd_list.append(opt_cvar_pd)
    opt_mv_pd_list.append(opt_mv_pd)

# Save the backtesting result
name = ["9302_0012", "0012_0502", "0502_1012", "1012_1912", "1912_2309"]
targ_vol_list = [0.15, 0.12, 0.1, 0.08, 0.05] / np.sqrt(12)
for i in range(5):
    for tv in targ_vol_list:
        opt_cvar_pd_list[i][opt_cvar_pd_list[i]["Target_vol"] == tv].to_csv("Output/Data/Backtest/cvar_" + name[i]
                                                                            + str(np.round(tv, 3)) +".csv")
        opt_mv_pd_list[i][opt_mv_pd_list[i]["Target_vol"] == tv].to_csv("Output/Data/Backtest/mv_" + name[i]
                                                                        + str(np.round(tv, 3))  + ".csv")

# Walk forward for 1/N naive portfolio
opt_1_over_N_pd_list = []
w = np.repeat(1 / 11, 11).T
for key in keys:
    r_i = r_ETF_train_test[key][key:]
    r_i = r_i @ w
    r_i = pd.DataFrame(r_i.values)
    opt_1_over_N_pd_list.append(r_i)

for i in range(5):
    opt_1_over_N_pd_list[i].to_csv("Output/Data/Backtest/naiev_" + name[i] + ".csv")

################## Don't run ####################################

# Part 2: 10 year simple walk forward
key = pd.to_datetime("2013-09-01")
opt_cvar_pd, opt_mv_pd = WalkForward.cv_once(r_ETF,
                    g,
                    pi, test_start=key)

for tv in targ_vol_list:
    opt_cvar_pd[opt_cvar_pd["Target_vol"] == tv].to_csv("Output/Data/Backtest/WalkForwad_10years/cvar_" + "130901_230901"
                                                                        + str(np.round(tv, 3)) +".csv")
    opt_mv_pd[opt_mv_pd["Target_vol"] == tv].to_csv("Output/Data/Backtest/WalkForwad_10years/mv_" + '130901_230901'
                                                                    + str(np.round(tv, 3))  + ".csv")

# Comments: All backtesting results are saved. Analysis are shown in another script.

