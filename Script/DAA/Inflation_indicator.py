#########################################
### Inflation indicator Demonstration ###
#########################################

import pandas as pd
import matplotlib.pyplot as plt
from Script.DAA import L1_filtration as l1

# UIG Data
UIG = pd.read_csv("Input/MACRO_DATA/UIGFULL.csv", index_col=0, parse_dates=True)
CPI = pd.read_csv("Input/MACRO_DATA/CORESTICKM159SFRBATL.csv", index_col=0, parse_dates=True)

UIG.columns = ["Inflation_indicator"]
CPI.columns = ["Inflation_indicator"]

# 1st date CPI available
_1st_date = UIG.index[0]
CPI_mimik = CPI[:_1st_date].iloc[:-1]


# stack together
pi_t = pd.concat([CPI_mimik, UIG], axis=0).sort_index()
pi_t = (pi_t - pi_t.mean()) / pi_t.std()
pi_t.to_csv("Output/DATA/inflation_indicator.csv")


# l1-filtration

# Entire history with l1-filtration
pi_t_l1 = pd.DataFrame(l1.l1_filter(pd.Series(pi_t["Inflation_indicator"]).values, 0.3).value)
pi_t_l1.index = pi_t.index
pi_t_l1.to_csv("Output/DATA/inflation_indicator_l1.csv")

