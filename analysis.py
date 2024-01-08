############################################################
### FIN 60202 Project 3: A CVaR optimization Approach to ###
###               ESG concerning ETF Allocation          ###
###                      ANALYSIS                        ###
############################################################

import datetime
import pandas as pd
import numpy as np
import Script.DataPrep.metric as mtc
import os
import glob
import re
import matplotlib.pyplot as plt

# targ vo
target_vol = [0.15, 0.12,0.1,0.08,0.05]

r_ETF = pd.read_csv("Output/DATA/ETFs_month_returns.csv", index_col=0, parse_dates=True)
r_ETF = r_ETF[:"2023-09-01"]
dateparse = lambda x: datetime.datetime.strptime(x, '%Y%m')
rf = pd.read_csv("Input/FF_MONTHLY.csv", index_col=0, parse_dates=True, date_parser=dateparse)/100

# Read in all data
df_dict = {}  # Create an empty dictionary
csv_files = glob.glob(os.path.join(os.getcwd() + "/Output/DATA/Backtest", "*.csv"))
for csv_f in csv_files:
    fname = os.path.basename(csv_f)  # Just the filename without directory prefix
    df = pd.read_csv(csv_f, index_col=0)
    df_dict[fname[:-4]] = df  # Add a dictionary entry where the key is <fname> and the value is <df>


# Assign time index
for i in df_dict.keys():
    num = re.findall(r'\d+', i)
    start = num[0]
    end = num[1][:-1]


    if start[0] == "9":
        start = "19" + start + "01"
    else:
        start = "20" + start + "01"

    end = "20" + end + "01"

    # to date time
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # assign index
    ind = pd.date_range(start=start, end=end, freq='MS')
    df_dict[i].index = ind

del csv_f, csv_files, df, end, fname, i, ind, num, start


# Sharp ratio
for tv in target_vol:
    l = mtc.get_model_data(df_dict, "mv", tv)
    l_sr = []
    for df in l:
        f = np.vectorize(mtc.sharp_ratio)

        # get r, rf and sigma
        r = df["Realized_r_t"]

        tp = [df.index[0], df.index[-1]]
        r_f = rf[tp[0] : tp[1]]
        r_f = list(r_f.values.T[0])

        t0 = df.index[0]
        tT = df.index[-1]
        cov_mat = mtc.get_hist_cov(t0, tT, r_ETF)
        sig_p = np.sqrt(np.diag(df.iloc[:,range(11)] @ cov_mat @ df.iloc[:, range(11)].T))
        l_sr += list(f(r, r_f, sig_p))

    # Annualized
    sr_annu = np.mean(l_sr) * 12
    print(str(tv) + " mv SR is" + str(sr_annu))


l = mtc.get_model_data(df_dict, "naiev", targ_vo = None)
l_sr = []
for df in l:
    f = np.vectorize(mtc.sharp_ratio)

    # get r, rf and sigma
    r = df.iloc[:,0]

    tp = [df.index[0], df.index[-1]]
    r_f = rf[tp[0] : tp[1]]
    r_f = list(r_f.values.T[0])

    t0 = df.index[0]
    tT = df.index[-1]
    cov_mat = mtc.get_hist_cov(t0, tT, r_ETF)
    sig_p = np.sqrt(np.repeat(1/11, 11).T @ cov_mat @ np.repeat(1/11, 11))

    l_sr += list(f(r, r_f, sig_p))

# Annualize
sr_annu = np.mean(l_sr) * 12
print(" naive SR is" + str(sr_annu))




# Turnover rate
mod = ["cvar", "mv"]
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        print(m)
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            tor = list(mtc.turnover_df(df.iloc[:,range(11)]))
            l_m += tor
        df_cvar_mv[m] = l_m

    plt.hist(df_cvar_mv.iloc[:,0], bins = 30, alpha=0.5,
             label = r"CVaR, $\mu =$"  + str(round(df_cvar_mv.mean()[0], 3)) +
             r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[0], 3)))
    plt.hist(df_cvar_mv.iloc[:,1],bins = 30, alpha=0.5,
             label = "MV, $\mu =$"  + str(round(df_cvar_mv.mean()[1], 3)) +
             r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[1], 3)))
    plt.title(r"Turnover Rate Distribution, $\sigma_{tar}$ = " + str(tv * 100) + "%")
    plt.legend()
    plt.gcf()
    # plt.savefig("Output/FIGURE/Results/TurnoverRate/tor_" + str(tv) + ".png", dpi = 300)
    plt.show()





# ESG scores related
ESG_Score = pd.read_csv("Output/DATA/ESG_Score.csv", index_col=0)
mod = ["cvar", "mv"]
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        print(m)
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            ESG_s = list((np.array(df.iloc[:,range(11)]) @ ESG_Score).T.values[0])
            l_m += ESG_s
        df_cvar_mv[m] = l_m
    plt.figure(figsize=(8,6))
    plt.hist(df_cvar_mv.iloc[:,0], bins = 30, alpha=0.7,
             label = r"CVaR, $\mu =$"  + str(round(df_cvar_mv.mean()[0], 3)) +
             r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[0], 3)), color="steelblue")
    plt.hist(df_cvar_mv.iloc[:,1],bins = 30, alpha=0.7,
             label = "MV, $\mu =$"  + str(round(df_cvar_mv.mean()[1], 3)) +
             r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[1], 3)), color="tab:green")
    plt.title(r"ESG Scores Distribution, $\sigma_{tar}$ = " + str(tv * 100) + "%")
    plt.legend()
    plt.gcf()
    plt.savefig("Output/FIGURE/Results/ESGScore/ESG_dist" + str(tv) + ".png", dpi = 300)
    plt.show()
# ESG WF overtime

# ESG scores related

# Read in all data
df_dict = {}  # Create an empty dictionary
csv_files = glob.glob(os.path.join(os.getcwd() + "/Output/DATA/Backtest/WalkForwad_10years", "*.csv"))
for csv_f in csv_files:
    fname = os.path.basename(csv_f)  # Just the filename without directory prefix
    df = pd.read_csv(csv_f, index_col=0)
    df_dict[fname[:-4]] = df  # Add a dictionary entry where the key is <fname> and the value is <df>


# Assign time index
for i in df_dict.keys():
    df_dict[i].index = pd.date_range(start=pd.to_datetime("2013-09-01")
                                     , end=pd.to_datetime("2023-09-01"), freq='MS')

mod = ["cvar", "mv"]
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            ESG_s = list((np.array(df.iloc[:,range(11)]) @ ESG_Score).T.values[0])
            l_m += ESG_s
        df_cvar_mv[m] = l_m
    df_cvar_mv.index = pd.date_range(start=pd.to_datetime("2013-09-01"),
                                     end=pd.to_datetime("2023-09-01"), freq='MS')
    plt.figure(figsize=(10, 3))
    plt.plot(df_cvar_mv.iloc[:, 0], label=r"CVaR, $\mu =$" + str(round(df_cvar_mv.mean()[0], 3)) +
                   r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[0], 3)), color="steelblue")
    plt.plot(df_cvar_mv.iloc[:, 1], label="MV, $\mu =$" + str(round(df_cvar_mv.mean()[1], 3)) +
                   r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[1], 3)), color="tab:green")
    plt.title(r"ESG Hypothetical Evolution, $\sigma_{tar}$ = " + str(tv * 100) + "%")
    plt.legend(bbox_to_anchor=(0.9, 1.1), loc="upper left", prop={'size':7})
    plt.tight_layout()
    plt.gcf()
    # plt.savefig("Output/FIGURE/Results/ESGScore/ESG" + str(tv) + ".png", dpi=300)
    plt.show()



# Read in all data
df_dict = {}  # Create an empty dictionary
csv_files = glob.glob(os.path.join(os.getcwd() + "/Output/DATA/Backtest", "*.csv"))
for csv_f in csv_files:
    fname = os.path.basename(csv_f)  # Just the filename without directory prefix
    df = pd.read_csv(csv_f, index_col=0)
    df_dict[fname[:-4]] = df  # Add a dictionary entry where the key is <fname> and the value is <df>


# Assign time index
for i in df_dict.keys():
    num = re.findall(r'\d+', i)
    start = num[0]
    end = num[1][:-1]


    if start[0] == "9":
        start = "19" + start + "01"
    else:
        start = "20" + start + "01"

    end = "20" + end + "01"

    # to date time
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # assign index
    ind = pd.date_range(start=start, end=end, freq='MS')
    df_dict[i].index = ind

# Fees
fees = pd.read_csv("Output/DATA/ETF_FEES.csv",index_col=0) * 100

mod = ["cvar", "mv"]
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            fee = list((np.array(df.iloc[:,range(11)]) @ fees).T.values[0])
            l_m += fee
        df_cvar_mv[m] = l_m
    plt.hist(df_cvar_mv.iloc[:,0], bins = 30, alpha=0.5,
             label = r"CVaR, $\mu =$"  + str(round(df_cvar_mv.mean()[0], 3)) +
             r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[0], 3)))
    plt.hist(df_cvar_mv.iloc[:,1],bins = 30, alpha=0.5,
             label = "MV, $\mu =$"  + str(round(df_cvar_mv.mean()[1], 3)) +
             r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[1], 3)))
    plt.title(r"Fees (%) Distribution, $\sigma_{tar}$ = " + str(tv * 100) + "%")
    plt.legend()
    plt.gcf()
    # plt.savefig("Output/FIGURE/Results/Fees/fee" + str(tv) + ".png", dpi = 300)
    plt.show()






# Accuracy MAPE for 3 models
for tv in target_vol:
    l = mtc.get_model_data(df_dict, "cvar", tv)
    l_sr = []
    for df in l:
        r_pred = df["mu"]
        r_realized = df["Realized_r_t"]
        l_sr += list(np.abs(r_pred - r_realized))
    mape = np.mean(l_sr)
    print(str(tv) + " cvar is" + str(mape))



l = mtc.get_model_data(df_dict, "naiev", targ_vo = None)
l_sr = []
for df in l:
    r_realized = df.iloc[:,0]

    tp = [df.index[0], df.index[-1]]
    r_hist = r_ETF[tp[0]:tp[1]]
    r_hist = r_hist.shift(1).iloc[1:]

    r_pred = list(r_hist @ np.repeat(1/11, 11))
    l_sr += list(np.abs(r_pred - r_realized[1:]))
mape = np.mean(l_sr)
print(str(tv) + " naiev mv is" + str(mape))



# Ask-bid transaction cost
ask_bid_percentage = [0.02, 0.02, 0.03, 0.09, 0.04, 0.03, 0.01, 0.02, 0.05, 0.01, 0.05]

mod = ["cvar", "mv"]
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            w = df.iloc[:,range(11)]
            delta_w = np.abs(w - w.shift(1))[1:]
            delta_w_normalize = delta_w.apply(lambda x: x / sum(x), axis = 1)

            abc = list((np.array(delta_w_normalize) @ ask_bid_percentage))
            l_m += abc
        df_cvar_mv[m] = l_m
    plt.hist(df_cvar_mv.iloc[:,0], bins = 30, alpha=0.5,
             label = r"CVaR, $\mu =$"  + str(round(df_cvar_mv.mean()[0], 3)) +
             r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[0], 5)))
    plt.hist(df_cvar_mv.iloc[:,1],bins = 30, alpha=0.5,
             label = "MV, $\mu =$"  + str(round(df_cvar_mv.mean()[1], 3)) +
             r", $\sigma^2 = $" + str(round(df_cvar_mv.var()[1], 5)))
    plt.title(r"Asd-bid Transaction Cost (%) Distribution, $\sigma_{tar}$ = " + str(tv * 100) + "%")
    plt.legend()
    plt.gcf()
    # plt.savefig("Output/FIGURE/Results/Abc/abc" + str(tv) + ".png", dpi = 300)
    plt.show()



# Information Ratio
# cvar vs mv
mod = ["cvar", "mv"]
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            r = df["Realized_r_t"]
            l_m +=list(r)
        df_cvar_mv[m] = l_m

    # Construt IR
    ri_rb = df_cvar_mv.iloc[:, 0] - df_cvar_mv.iloc[:,1]
    sigma_TE = np.std(ri_rb)
    IR = np.mean(ri_rb) / sigma_TE
    print("Cvar vs MV IR: targvol = " + str(tv) + " = " + str(IR) )


# cvar vs naiev
mod = ["cvar", "naiev"]
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            if m == "cvar":
                r = df["Realized_r_t"]
            else:
                r = df.iloc[:,0]
            l_m +=list(r)
        df_cvar_mv[m] = l_m

    # Construt IR
    ri_rb = df_cvar_mv.iloc[:, 0] - df_cvar_mv.iloc[:,1]
    sigma_TE = np.std(ri_rb)
    IR = np.mean(ri_rb) / sigma_TE
    print("Cvar vs naiev IR: targvol = " + str(tv) + " = " + str(IR) )


# MV vs naiev
mod = ["mv", "naiev"]
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            if m == "mv":
                r = df["Realized_r_t"]
            else:
                r = df.iloc[:,0]
            l_m +=list(r)
        df_cvar_mv[m] = l_m

    # Construt IR
    ri_rb = df_cvar_mv.iloc[:, 0] - df_cvar_mv.iloc[:,1]
    sigma_TE = np.std(ri_rb)
    IR = np.mean(ri_rb) / sigma_TE
    print("MV vs naiev IR: targvol = " + str(tv) + " = " + str(IR) )


# Hypothetical Return (10 year walk forward)


# Read in all data
df_dict = {}  # Create an empty dictionary
csv_files = glob.glob(os.path.join(os.getcwd() + "/Output/DATA/Backtest/WalkForwad_10years", "*.csv"))
for csv_f in csv_files:
    fname = os.path.basename(csv_f)  # Just the filename without directory prefix
    df = pd.read_csv(csv_f, index_col=0)
    df_dict[fname[:-4]] = df  # Add a dictionary entry where the key is <fname> and the value is <df>


# Assign time index
for i in df_dict.keys():
    df_dict[i].index = pd.date_range(start=pd.to_datetime("2013-09-01")
                                     , end=pd.to_datetime("2023-09-01"), freq='MS')

# Cvar vs mv
mod = ["cvar", "mv"]
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            cum_r = (df["Realized_r_t"] + 1).cumprod()
            l_m += list(cum_r)
        df_cvar_mv[m] = l_m
    df_cvar_mv.index = pd.date_range(start=pd.to_datetime("2013-09-01"),
                                     end=pd.to_datetime("2023-09-01"), freq='MS')

    # Get 1year, 3 year, 5 year and 10 year annual return
    r_1year = df_cvar_mv.loc["2023-09-01"] / df_cvar_mv.loc["2022-09-01"]  - 1
    r_3year = (df_cvar_mv.loc["2023-09-01"] / df_cvar_mv.loc["2020-09-01"])**(1/3) - 1
    r_5year = (df_cvar_mv.loc["2023-09-01"] /df_cvar_mv.loc["2018-09-01"])**(1/5) - 1
    r_10year = (df_cvar_mv.loc["2023-09-01"] /df_cvar_mv.loc["2013-09-01"])**(1/10) - 1
    print("Target volatility = " + str(tv) +", 1 year return: CVaR " + str(round(r_1year["cvar"], 5)) + ": "  + "MV : " + str(round(r_1year["mv"], 5)))
    print("Target volatility = " + str(tv) +", 3 year return: CVaR " + str(round(r_3year["cvar"], 5)) + ": " + "MV : " + str(round(r_3year["mv"], 5)))
    print("Target volatility = " + str(tv) +", 5 year return: CVaR " + str(round(r_5year["cvar"], 5)) + ": " + "MV : " + str(round(r_5year["mv"], 5)))
    print("Target volatility = " + str(tv) +", 10 year return: CVaR " + str(round(r_10year["cvar"], 5)) + ": " + "MV : " + str(round(r_10year["mv"], 5)))

    plt.figure(figsize=(10, 3))
    plt.plot(df_cvar_mv.iloc[:, 0], label=r"CVaR", color="steelblue")
    plt.plot(df_cvar_mv.iloc[:, 1], label="MV", color="tab:green")
    plt.title(r"10 Year Hypothetical Return, $\sigma_{tar}$ = " + str(tv * 100) + "%")
    plt.legend(loc="upper left", prop={'size':9})
    plt.tight_layout()
    plt.gcf()
    # plt.savefig("Output/FIGURE/Results/return/" + str(tv) + ".png", dpi=300)
    plt.show()



# CVarR vs 1/N
mod = ["cvar"]
plt.figure(figsize=(15,8))
for tv in target_vol:
    df_cvar_mv = pd.DataFrame(columns=mod)
    for m in mod:
        l = mtc.get_model_data(df_dict, m, tv)
        l_m = []
        for df in l:
            cum_r = (df["Realized_r_t"] + 1).cumprod()
            l_m += list(cum_r)
        df_cvar_mv[m] = l_m
    df_cvar_mv.index = pd.date_range(start=pd.to_datetime("2013-09-01"),
                                     end=pd.to_datetime("2023-09-01"), freq='MS')

    # Get 1year, 3 year, 5 year and 10 year annual return

    plt.plot(df_cvar_mv.iloc[:, 0], label=str(tv * 100) + "%")
    plt.title(r"10 Year Hypothetical Return: CVaR v.s. 1/N", fontsize = 20)

tp = [df.index[0], df.index[-1]]
r_hist = r_ETF[tp[0]:tp[1]]
r_pred = r_hist @ np.repeat(1/11, 11)
r_cum = (1 +r_pred).cumprod()

r_1year = r_cum.loc["2023-09-01"] / r_cum.loc["2022-09-01"]  - 1
r_3year = (r_cum.loc["2023-09-01"] / r_cum.loc["2020-09-01"])**(1/3) - 1
r_5year = (r_cum.loc["2023-09-01"] /r_cum.loc["2018-09-01"])**(1/5) - 1
r_10year = (r_cum.loc["2023-09-01"] /r_cum.loc["2013-09-01"])**(1/10) - 1
print("1/N" + " ,1 year return: CVaR " + str(round(r_1year, 5)))
print("1/N" + ", 3 year return: CVaR " + str(round(r_3year, 5)))
print("1/N" +  ", 5 year return: CVaR " + str(round(r_5year, 5)))
print("1/N" + ", 10 year return: CVaR " + str(round(r_10year, 5)))



plt.plot(r_cum, label = "1/N", linestyle = "dashdot",  linewidth = 3, c = "red")
plt.xticks(fontsize=14)
plt.yticks(fontsize = 14)
plt.legend(loc="upper left", prop={'size': 18})
plt.tight_layout()
plt.gcf()
# plt.savefig("Output/FIGURE/Results/return/10_year_WF" + str(tv) + ".png", dpi=300)
plt.show()

















