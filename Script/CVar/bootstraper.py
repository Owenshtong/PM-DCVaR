#########################################
###### Bootstrap Returns Factory ########
#########################################

import pandas as pd
import numpy as np
import random


def bs_Non_parametric(r_df, N = 2000, transform = "mean"):
    """
    :param r_df: Monthly return dataframe
    :param N:Monthly return dataframe
    :param transform: Function applied to each bs sample set. Could reasonably be min.
    :return: bootstrapped returns, bootstrapped covariance
    """
    n_row = r_df.shape[0]

    # Generate BS sample
    r_bs = pd.DataFrame(columns=r_df.columns, index = range(N))

    cov_bs = 0
    for i in range(N):
        ind_row = random.choices(range(n_row), k=n_row)
        bs_temp = r_df.iloc[ind_row]
        cov_bs += bs_temp.cov()

        t = bs_temp.mean()

        if transform == "min": # More extreme cases
            t = bs_temp.min()

        r_bs.iloc[i] = t

    cov_bs = cov_bs / N

    return r_bs, cov_bs


def bs_parametric_MVN(r_df, N = 100):
    """
    Fit a multivariate normal distribution based on r_df. Then bootstrap from the
    distribution N times
    :param r_df: Monthly return dataframe
    :param N: Number of bs samples
    :return:
    """

    # Parameter
    mu = r_df.mean()
    Sigma = r_df.cov()

    # Bootstrap
    boost_mvn = pd.DataFrame(
        np.random.multivariate_normal(
        mu, Sigma, N
    ))
    boost_mvn.columns = r_df.columns
    boost_cov = boost_mvn.cov()
    return boost_mvn, boost_cov


