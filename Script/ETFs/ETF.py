#######################################################
################# ETF Object builder ##################
#######################################################

import pandas as pd
import yfinance as yf
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
import pandas_datareader as pdr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class ETF:

    def __init__(self, ticker, method, bench_ticker = "^GSPC", bench_source = "yf", bench_csv = None):
        """

        :param ticker: ETF ticker
        :param bench_ticker: The benchmark index used to mimic history
        :param bench_source: in ["yf", "pdr", "csv]. If csv then bench_csv has to be given as input
        """
        self.bench_source = bench_source
        self.bench_ticker = bench_ticker
        self.bench_csv = bench_csv
        self.ticker = ticker
        self.price_monthly= self.format_price(ticker, interval = "1mo")
        self.price_daily = self.format_price(ticker, interval= "1d")
        self.market_cap = None  # Dollar value of the ETF

        self.ESG_score = None
        self.expense_ratio = None

        self.r_monthly = self.returns(self.price_monthly, "r_month")
        self.r_daily = self.returns(self.price_daily, "r_day")
        self.benchmark = None

        self.mimic_method = method
        if ticker == bench_ticker:
            self.r_mimic_hist = None
        if bench_source == "yf":
            self.r_mimic_hist = self.Mimic_history(y_ticker=bench_ticker)
        if bench_source == "pdr":
            self.r_mimic_hist = self.Mimic_history(y_ticker=bench_ticker)
        if bench_source == "csv":
            self.r_mimic_hist = self.Mimic_history(y_ticker=bench_ticker)




    @staticmethod
    def format_price(ticker, interval):
        hist = yf.download(ticker,interval=interval)
        hist.reset_index(inplace=True)
        hist.index = pd.to_datetime(hist["Date"].dt.strftime('%Y/%m/%d'))
        hist = hist.drop("Date", axis = 1)
        return pd.DataFrame(hist)

    @staticmethod
    def returns(price_data, col_name):
        """
        :return: Notice the index t means the return of period t. For example
        2017-02-01 is the return from 2017-02-01 to 2017-02-30. The day -01 doesn't matter. Since the price data are all
        obtained on 1st date of the month. It's like an annuity-due. This is also consistent
        with the index in Macro data obtained from Dynamic asset allocation.
        """
        hist = copy.copy(price_data)
        hist[col_name] = (hist["Adj Close"] -
                          hist["Adj Close"].shift(1)) / hist["Adj Close"].shift(1)
        hist = hist[col_name]
        hist = hist.iloc[1:-1]
        return pd.DataFrame(hist)

    def get_inception_ym(self):
        """
        :return: The first date with available returns
        """
        return self.r_monthly.index[0]


    def Mimic_history(self, y_ticker):
        """
        Regress ETF return on a market/benchmark returns. To mimic historical return of ETF
        :param: method: if regression using OLS, if skl using sklearn blackbox imputation
        :return: Mimicked historical return + available actual returns
        """
        # First date of ETF return observation
        co_start = self.get_inception_ym()

        if self.mimic_method == "regression":
            if self.bench_source == "yf":
                # Benchmark returns (usually market returns e.g. sp500)
                anchor_r_month = self.returns(
                    self.format_price(ticker = y_ticker ,interval = "1mo"),
                    col_name = "r_month"
                )
                y = self.r_monthly

            if self.bench_source == "pdr":
                anchor_r_month = pdr.get_data_fred(self.bench_ticker, start = "1980-01-01") / 100
                y = self.r_monthly

            if self.bench_source == "csv":
                anchor_r_month = self.bench_csv
                co_end = min(anchor_r_month.index[-1], self.r_monthly.index[-1])
                anchor_r_month = anchor_r_month[:co_end]
                print(anchor_r_month)
                y = self.r_monthly[:co_end]


            X = anchor_r_month[co_start:]


            # OLS
            model = LinearRegression()
            model.fit(X, y)

            # Generate history
            X_pred = anchor_r_month[~anchor_r_month.index.isin(X.index)]
            y_pred = pd.DataFrame(model.predict(X_pred))
            y_pred.index = X_pred.index
            y_pred = y_pred.rename(columns={0: y.columns[0]})

            r_month_mimic = pd.concat([y_pred, y], axis=0)

        if self.mimic_method == "SKL":
            if self.bench_source == "yf":
                # Benchmark returns (usually market returns e.g. sp500)
                anchor_r_month = self.returns(
                    self.format_price(ticker = y_ticker, interval = "1mo"),
                    col_name = "r_month"
                )

            if self.bench_source == "pdr":
                anchor_r_month = pdr.get_data_fred("GS3",start = "1980-01-01") / 100

            if self.bench_source == "csv":
                anchor_r_month = self.bench_csv

            pds = pd.merge(self.r_monthly, anchor_r_month, left_index=True, right_index=True, how = "outer")
            pds_impute = IterativeImputer(random_state=0)
            pds_impute.fit(pds)

            r_month_mimic = pd.DataFrame(pds_impute.transform(pds), columns=["r_month", "r_bench"])
            r_month_mimic = r_month_mimic["r_month"]
            r_month_mimic.index = pds.index
            r_month_mimic.index.name = "Date"


        return r_month_mimic

        

# TODO: For the regression, may save the regression object and quickly look at its
#       significance.


















