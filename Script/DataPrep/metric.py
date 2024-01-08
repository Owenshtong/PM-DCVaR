########################################################################
############### Functions to compute Performance metric ################
########################################################################

import datetime
import pandas as pd
import numpy as np

def sharp_ratio(r, rf, vol_p):
    return  (r - rf)/vol_p

def get_model_data(dictionary, model, targ_vo):
    """
    given a dictionay, get a list of df of models
    :param dictionary: model = cvar, mv, naiev
    :param targ_vo: targ_vol = 0.15, 0.12, 0.10, 0.08, 0.05
    :param cv: the cv period
    :return:
    """
    if targ_vo == 0.15:
        vol_suf = ".043"
    if targ_vo == 0.12:
        vol_suf = ".035"
    if targ_vo == 0.10:
        vol_suf = ".029"
    if targ_vo == 0.08:
        vol_suf = ".023"
    if targ_vo == 0.05:
        vol_suf = ".014"

    # Filter
    l = []
    if model == "naiev":
        for i in dictionary.keys():
            if model in i:
                l.append(dictionary[i])

    else:
        for i in dictionary.keys():
            if (model in i) & (vol_suf in i):
                l.append(dictionary[i])

    return l


def get_hist_cov(start, end, r_ETF):
    """
    return the
    :param t_start:
    :param t_end:
    :param r_ETF:
    :return:
    """
    ind = pd.date_range(start=start, end=end, freq='MS')
    df_train = r_ETF[~np.isin(r_ETF.index, ind)]

    covmat = df_train.cov()
    return covmat

def turnover_df(df):
    """
    take a df of weight over time. Reuturn the list of turnover rate
    :return:
    """

    ddf = abs(df - df.shift(1)).apply(lambda x: sum(x), axis = 1)[1:]

    return ddf