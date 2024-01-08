#########################################
#### Growth indicator Demonstration #####
#########################################

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Script.ETFs import  ETF
from Script.DAA import L1_filtration as l1


### Growth Indicator ###

END_DATE = "2023-10-01"

TYFF =  pd.read_csv("INPUT/MACRO_DATA/T10YFFM.csv", index_col=0)
BAA = pd.read_csv("INPUT/MACRO_DATA/BAA10YM.csv", index_col=0)
JOB = pd.read_csv("INPUT/MACRO_DATA/IC4WSA.csv", index_col=0)
PERMIT = pd.read_csv("INPUT/MACRO_DATA/PERMIT.csv", index_col=0)
VIX = pd.read_csv("INPUT/MACRO_DATA/VIXCLS.csv", index_col=0)

growth_vars = pd.concat([TYFF,BAA,JOB,PERMIT,VIX], axis=1).sort_index()
growth_vars.columns = ["TYFF", "BAA", "JOB", "PERMIT", "VIX"]
growth_vars = growth_vars.loc["1976-01-01":]

# Proxy VIX by SP500 sd from 1986-01-01 t0 1989-12-01
SP500 = ETF.ETF("^GSPC")
SP500.price_daily = SP500.price_daily.loc["1975/12/01" : "1989/12/01"]
SP500_MONTH_STD = SP500.price_daily.groupby(pd.Grouper(freq='M'))["Adj Close"].std()
SP500_MONTH_STD.index = SP500_MONTH_STD.index.shift(periods=1, freq='D')
SP500_MONTH_STD = SP500_MONTH_STD.drop("1990-01-01")
SP500_MONTH_STD = pd.DataFrame(SP500_MONTH_STD)
SP500_MONTH_STD.columns = ['VIXCLS']


# Merge VIX with SP500_MONTH_STD
VIX = pd.concat([SP500_MONTH_STD,VIX], axis = 0)
growth_vars["VIX"] = list(VIX["VIXCLS"])

# Standardize each variable (Page 138 Kim & Kwon (2023))
growth_vars = (growth_vars - growth_vars.mean()) / growth_vars.std()

# Except for PERMIT, inverted by * (-1) due to counter-cyclical effect
col_name_counter = list(growth_vars.columns.values)
col_name_counter.remove("PERMIT")
for i in col_name_counter:
    growth_vars[i] = growth_vars[i] * (-1)


# Cutoff at END_DATE
growth_vars = growth_vars.loc[:END_DATE]

# One lag for JOB and PERMIT (Page 138 foot note Kim & Kwon (2023))
lag = 1
growth_vars["JOB_1"] = growth_vars["JOB"].shift(lag)
growth_vars["PERMIT_1"] = growth_vars["PERMIT"].shift(lag)
growth_vars = growth_vars.iloc[lag:]
growth_vars = growth_vars.drop(columns = ["JOB", "PERMIT"])

# PCA over adjusted variables from above
pca = PCA(n_components=3) # 3 principle. Only the 1st one is needed
pca.fit(growth_vars)

# Info of PCA
var_explained_ratio = pca.explained_variance_ratio_
pc1 = pca.components_[0,:]

# Growth indicator
gt = (-1) * growth_vars @ pc1.T
gt.to_csv("Output/DATA/growth_indicator.csv")


# l1-filtration
gt_l1 = pd.DataFrame(l1.l1_filter(gt.values, 0.3).value)
gt_l1.index = gt.index
gt_l1.to_csv("Output/DATA/growth_indicator_l1.csv")


# TODO: When building function for backtesting, the regime is forecasted one-month ahead
#      See footnote on page 141. The plot GrowthIndicator_pc1(_l1).png is plotted
#      standing at 2022.
