#############################################
### Functions to be used related to DAA #####
#############################################
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


def indicator_forecast(series):
    """
    1-step-ahead out-of-sample prediction. Used for inflation and growth indicator
    :param series: A pd.Series object
    :return:
    """
    mod = AutoReg(series, 1, old_names=False).fit()
    pred_1_period = mod.predict(start = len(series), end = len(series)).values[0]

    return pred_1_period

def classify_regime(gt_Series, pit_Series, ETF_r_monthly):
    """
    Should be applied to a pd.dataframe containing growth and inflation indicators. Assign values
    as follows:
                    Inflation
                    +                -
        Growth  +   Heating_up      Goldilocks

                -   Slow_growth     Stagflation
    :return: pd.Dataframe with 1 column, showing which regime each row belongs to
    """
    g = pd.DataFrame(gt_Series)
    pi = pd.DataFrame(pit_Series)
    ind = pd.merge(g, pi, left_index=True, right_index=True)
    ind.columns = ["g", "pi"]

    # Fist difference
    ind_diff = ind.diff(1)

    # Assign regime
    ind_diff = ind_diff.iloc[1:]
    ind_diff["regime"] =  ind_diff.apply(lambda x: classify_1lag(x["g"], x["pi"]),
                                         axis=1)
    regime = pd.DataFrame(ind_diff["regime"])

    # Merge with ETF monthly returns
    df = pd.merge(ETF_r_monthly, regime, left_index=True, right_index=True)

    return df

def classify_1lag(g, pi):
    """
    Assign regime to each historical row
    :param g: growth indicator
    :param pi: inflation indicator
    :return:
    """
    rgm = "None"
    if (g > 0) & (pi > 0):
        rgm = "Heating_up"
    if (g > 0) & (pi < 0):
        rgm = "Goldilocks"
    if (g < 0) & (pi > 0):
        rgm = "Slow_growth"
    if (g < 0) & (pi < 0):
        rgm = "Stagflation"
    return rgm


def classify_1lag_vect(g, pi):
    """
    Vectorized version of classify_1lag
    :param g: growth indicator
    :param pi: inflation indicator
    :return:
    """
    fun = np.vectorize(classify_1lag)
    return fun(g, pi)


def get_regime_return(df, reg):
    """
    :param df: Monthly return with regime indicator as a column.
    :param reg: Regime, str.
    :return: Monthly return of the given regime.
    """
    regime_r = df[
        df["regime"] == reg
    ]

    regime_r = regime_r.drop("regime", axis = 1)
    regime_r = regime_r.reset_index(drop = True)
    return regime_r


# TODO: Mentioned that one of the reason we consider a parametric approach is that
#       based on the model, there are much more Goldilocks and Heating_up states than Slow_growth
#       and Stagflation.



