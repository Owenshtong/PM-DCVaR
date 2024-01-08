################################################
### Some discriptive plots can be added here ###
################################################

import pandas as pd
import matplotlib.pyplot as plt
from Script.DAA.functions import classify_regime


### Plot 1: distribution of ETFs in different regime

pi_t_l1 = pd.read_csv("Output/DATA/inflation_indicator_l1.csv", index_col=0, parse_dates=True)
g_t_l1 = pd.read_csv("Output/DATA/growth_indicator_l1.csv", index_col=0, parse_dates=True)
ETF_monthly = pd.read_csv("Output/DATA/ETFs_month_returns.csv", index_col=0, parse_dates=True)

# ETF_monthly = ETF_monthly[:"2000-12-01"]
pi_t_l1 = pd.Series(pi_t_l1["0"])
g_t_l1 = pd.Series(g_t_l1["0"])

reg_r = classify_regime(g_t_l1, pi_t_l1, ETF_monthly)
reg_r = reg_r.reset_index(drop = True)
reg_r["id"] = reg_r.index

reg_r_pivot = reg_r.melt(id_vars=['regime'], value_vars = reg_r.columns[:-1])

reg_r_pivot.hist('value', by=["variable", 'regime'],
                sharex=False, sharey=False,
                layout=(4,11), figsize=(30,20), xrot=0, bins = 30)
plt.gcf()
plt.savefig("Output/FIGURE/ETF_r_by_Regime.png")
plt.show()



### Plot 2: Economic regime indicator
g_t = pd.read_csv("Output/DATA/Growth_indicator.csv", index_col=0, parse_dates=True)
pi_t = pd.read_csv("Output/DATA/inflation_indicator.csv", index_col=0, parse_dates=True)



## Inflation indicator
fig, ax = plt.subplots(figsize=[12, 6])
ax.plot(pi_t, linewidth = 3 ,label = r"Inflation Indicator $g_t$")


# Recessions identified by NBER are shaded
alpha = 0.3
ax.fill_between(["1980-02-01", "1980-06-01"], -2, 5, color='gray', alpha=alpha, label='Recession (NBER)')
ax.fill_between(["1981-07-01", "1982-11-01"], -2, 5, color='gray', alpha=alpha)
ax.fill_between(["1990-07-01", "1991-03-01"], -2, 5, color='gray', alpha=alpha)
ax.fill_between(["2001-04-01", "2001-11-01"], -2, 5, color='gray', alpha=alpha)
ax.fill_between(["2007-12-01", "2009-06-01"], -2, 5, color='gray', alpha=alpha)
ax.tick_params(labelsize=10)
plt.title(r"Inflation Indicator $\pi_t$ based on UIG and CPI")
plt.ylabel("Inflation indicator")
plt.tight_layout()
plt.legend(loc='upper left', fontsize=12)
plt.gcf()
plt.savefig("OUTPUT/FIGURE/InflationIndicator.png", dpi = 500)
plt.show()



## Inflation indicator-l1

# Plot to check
fig, ax = plt.subplots(figsize=[12, 6])
ax.plot(pi_t, label = r"Growth Indicator $g_t$", linewidth = 3)
ax.plot(pi_t_l1, label = r"$\ell_1$ Filtered Inflation Indicator",  color = "red", linewidth = 3)


# Recessions identified by NBER are shaded
ax.fill_between(["1980-02-01", "1980-06-01"], -2, 5, color='gray', alpha=alpha,  label='Recession (NBER)')
ax.fill_between(["1981-07-01", "1982-11-01"], -2, 5, color='gray', alpha=alpha)
ax.fill_between(["1990-07-01", "1991-03-01"], -2, 5, color='gray', alpha=alpha)
ax.fill_between(["2001-04-01", "2001-11-01"], -2, 5, color='gray', alpha=alpha)
ax.fill_between(["2007-12-01", "2009-06-01"], -2, 5, color='gray', alpha=alpha)
ax.tick_params(labelsize=7)
plt.title(r"Inflation Indicator $\pi_t$ based on UIG and CPI")
plt.ylabel("Inflation indicator")
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.gcf()
plt.savefig("OUTPUT/FIGURE/InflationIndicator_pc1_l1.png", dpi = 500)
plt.show()



## Growth indicator
fig, ax = plt.subplots(figsize=[12, 6])
ax.plot(g_t, linewidth = 3 ,label = r"Growth Indicator $g_t$")


alpha = 0.3
ax.fill_between(["1980-02-01", "1980-06-01"], -7, 7, color='gray', alpha=alpha, label='Recession (NBER)')
ax.fill_between(["1981-07-01", "1982-11-01"], -7, 7, color='gray', alpha=alpha)
ax.fill_between(["1990-07-01", "1991-03-01"], -7, 7, color='gray', alpha=alpha)
ax.fill_between(["2001-04-01", "2001-11-01"], -7, 7, color='gray', alpha=alpha)
ax.fill_between(["2007-12-01", "2009-06-01"], -7, 7, color='gray', alpha=alpha)
ax.tick_params(labelsize=7)
plt.title(r"Growth Indicator $g_t$ based on $1_{st}$ PC")
plt.ylabel("Growth indicator")
plt.tight_layout()
plt.legend(loc='upper left', fontsize=12)
plt.gcf()
plt.savefig("OUTPUT/FIGURE/GrowthIndicator_pc1.png", dpi = 500)
plt.show()




# Growth indicator-l1
# Plot to check
fig, ax = plt.subplots(figsize=[12, 6])
ax.plot(g_t, label = r"Growth Indicator $g_t$", linewidth = 3)
ax.plot(g_t_l1, label = r"$\ell_1$ Filtered Growth Indicator",  color = "red", linewidth = 3)


# Recessions identified by NBER are shaded
ax.fill_between(["1980-02-01", "1980-06-01"], -7, 7, color='gray', alpha=alpha)
ax.fill_between(["1981-07-01", "1982-11-01"], -7, 7, color='gray', alpha=alpha)
ax.fill_between(["1990-07-01", "1991-03-01"], -7, 7, color='gray', alpha=alpha)
ax.fill_between(["2001-04-01", "2001-11-01"], -7, 7, color='gray', alpha=alpha)
ax.fill_between(["2007-12-01", "2009-06-01"], -7, 7, color='gray', alpha=alpha)
ax.fill_between(["2020-02-01", "2020-04-01"], -7, 7, color='gray', alpha=alpha, label='Recession (NBER)')
ax.tick_params(labelsize=7)
plt.title(r"Growth Indicator $g_t$ based on $1_{st}$ PC")
plt.ylabel("Growth indicator")
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.gcf()
plt.savefig("OUTPUT/FIGURE/GrowthIndicator_pc1_l1.png", dpi = 500)
plt.show()
