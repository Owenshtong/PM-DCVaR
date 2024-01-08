#######################################################
###### Script to run in before main.py analysis ######
#######################################################

# Output:
#      - ETFs (Dictionary)
#      - ETFs monthly return each from its own inception (DataFrames)
#-------------------------------------------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt
from Script.ETFs.ETF import ETF



### Part1: ETFs Original Returns ###

# ESG and FEEs (expense ratio)
ESG_FEE = pd.read_excel("INPUT/Fees_ESG_score.xlsx")
ESG_fee = ESG_FEE["Expense ratio(%)"] / 100
ESG_fee.to_csv("Output/DATA/ETF_FEES.csv")
# csv source needed: The Aaa moody's  Corporate Bond Yield
Aaa = pd.read_csv("Input/AAA.csv", index_col=0, parse_dates=True)  / 100

# Tickers
tickers = list(ESG_FEE["Ticker"])
fee = list(ESG_FEE["Expense ratio(%)"])
ESG_score = ESG_FEE["MSCI ESG Quality Score (0-10)"]
ESG_score.to_csv("Output/DATA/ESG_Score.csv")
ESG_score = list(ESG_FEE["MSCI ESG Quality Score (0-10)"])
ESG_bench_ticker = list(ESG_FEE["Benchmark Index"])
ESG_bench_source = list(ESG_FEE["Source"])

# ETF Objects
ETFs = {}
for i in tickers:
    row_loc = tickers.index(i)
    ETFs[i] = ETF(i, bench_ticker=ESG_bench_ticker[row_loc],
                  bench_source=ESG_bench_source[row_loc],
                  method="SKL", bench_csv=Aaa)
    ETFs[i].ESG_score = ESG_score[row_loc]
    ETFs[i].expense_ratio = fee[row_loc]


# Plot the return series we imputed
plt.figure(figsize=(15, 8))
for i in tickers:
    row_loc = tickers.index(i)
    plt.plot(ETFs[i].r_mimic_hist, label = ETFs[i].ticker + ": " + ESG_bench_ticker[row_loc] + " Adjusted")
    plt.legend()
plt.title("Imputed Return Series for Each ETF", fontsize = 20)
plt.tight_layout()
plt.savefig("Output/FIGURE/Imputed_r.png")
plt.show()





# Sheets of Date, Ticker, Monthly (with mimicked historical returns)
ETFs_month_returns = ETFs[tickers[0]].r_mimic_hist
for i in range(len(tickers) - 1):
    tick_l = tickers[i]
    tick_r = tickers[i + 1]
    ETFs_month_returns = pd.merge(ETFs_month_returns,
                 ETFs[tickers[i+1]].r_mimic_hist,
                 how = "outer",
                 on = "Date",
                 suffixes = ("_" + tick_l, "_" + tick_r)
                 )
    ETFs_month_returns = ETFs_month_returns.sort_index()
ETFs_month_returns.columns = tickers
ETFs_month_returns = ETFs_month_returns["1993-02-01":]

# Save ETFs_month_returns
ETFs_month_returns.to_csv("OUTPUT/DATA/ETFs_month_returns.csv")

# Relex memory
del ESG_FEE, tickers, fee, ESG_score, i, row_loc, tick_r, tick_l