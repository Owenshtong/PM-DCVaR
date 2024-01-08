#############################################
##### A Walk-forward backtesting Scheme #####
#############################################

import copy
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from Script.DAA import functions as DAA
from Script.DAA.L1_filtration import l1_filter
from Script.CVar.CVaR_optimizer import CVar
from Script.CVar.bootstraper import bs_parametric_MVN, bs_Non_parametric
from Script.MinVar.MinVar_optimizer import MinVar


def get_hist(t, history_data):
    """
    Want to re-balance using data until (t-1) (included).
    :param t: The point of time to be forecasted. Should be of type datetime.
    :param history_data: The ENTIRE historical data. Should be a pd.dataframe
    :return:
    """

    t_1 = t - relativedelta(months=1)

    return history_data[:t_1]


def Walk_forward_CVaR(t, g, pi, r_ETF, targ_vol, lambda0 = 0.3,
                      bs_method = "nonparametric", N_mu_grid = 20, CL = 0.9,
                      ESG_cons = 6, target_vol_bl = False, strict_or_not = False):
    """
    The function to run at EACH t for optimization
    :param target_vol_bl: If target volatility constrain is active
    :param strict_or_not: If the target volatility constrain is strictly =
    :param r_ETF: entire history of ETF monthly returns
    :param t: Current point of time
    :param g: entire history growth indicator
    :param pi: entire history inflation indicator
    :param lambda0: The l1 smooth parameter
    :param bs_method: If nonparametric, use non-parametric bs. If parametric, use MVN parametric.
    :param N_mu_grid: Number of grid of mu (frontier)
    :param CL: Confidence level for CVaR optimization
    :param targ_vol: MONTHLY target volatility (max monthly vol can bear).
    :param ESG_cons: Minimum ESG score the portfolio should have.
    :return:
    """

    # Get history data
    g_hist = get_hist(t, g)
    pit_hist = get_hist(t, pi)
    r_ETF_hist = DAA.classify_regime(g_hist, pit_hist, get_hist(t, r_ETF))

    # l1_filtration on g and pi #
    g_hist_l1 = pd.Series(l1_filter(list(g_hist), lbd=lambda0),index=g_hist.index)
    pit_hist_l1 = pd.Series(l1_filter(list(pit_hist), lbd=lambda0), index=g_hist.index)

    # Forecast macro
    g_t = DAA.indicator_forecast(g_hist_l1)
    delta_g_t = g_t - g_hist.iloc[-1]
    pi_t = DAA.indicator_forecast(pit_hist_l1) # TODO: check the paper to see should be subtract l1 or original g and pi
    delta_pi_t = pi_t - pit_hist.iloc[-1]

    # Forcast regime
    regime_t = DAA.classify_1lag(delta_g_t, delta_pi_t)

    # Get historical returns for regime_t
    regime_t_ETF_r = DAA.get_regime_return(r_ETF_hist, regime_t)

    # Bootstrap from regime_t_ETF_r
    if bs_method == "nonparametric":
        bs_r, bs_cov = bs_Non_parametric(regime_t_ETF_r)
    if bs_method == "parametric":
        bs_r, bs_cov = bs_parametric_MVN(regime_t_ETF_r)


    ######################################## Models ##########################################
    # "Population" mean and var
    mu = bs_r.mean()
    Sigma = bs_r.cov()

    mu_space = np.linspace(mu[mu > 0].min(), max(mu) * 0.99, N_mu_grid)


    #### Model 1: CVaR Optimization: Over linespace of mus. ####
    # Logic: The target volatility can be too high to achieve. Due to no short-selling constrain, the max volatility
    #       can be met is the max volatility of individual assets. However, 100% in that asset might not be optimal
    #       for a CVaR objective function. But we are sure about one thing: higher expected return <--> higher sigma.
    #       Thus we do the following: (1) Grid mu from [mu.min, mu.max] (2) run optimization for each grid point (3)
    #       (3) Get the correspondence sigma sigma_max of mu.max, if sigma_max < target volatility, then the optimal
    #       weight is the one corresponds to sigma_max. If sigma_max > target volatility, rerun the optimization and
    #       set the strict = True, and assign mu = 0 to get the weight corresponding the target.

    cname = list(r_ETF.columns)
    cname.append("mu")
    cname.append("sigma_p") # portfolio sigma
    cname.append("CVaR")
    cname.append("Var")
    cname.append("Realized_r_t")
    cname.append("Target_vol")
    w_pd_CVaR = pd.DataFrame(columns= cname)


    for i in range(mu_space.shape[0]):
        m = mu_space[i]
        cvar, w, var, _ = CVar(beta=CL, R=m, returns_mat = bs_r, target_vol = targ_vol, mu=mu, Sigma=Sigma,
                               ESG_lb=ESG_cons, include_ESG_rating_target = True,
                               include_target_vol = target_vol_bl, strict = strict_or_not)

        if cvar is None:
            cvar, w, var = cvar_keep, w_keep, var_keep
            break

        mu_0 = w @ mu
        sigma = np.sqrt(w @ Sigma @ w)

        # Return realization
        r_bt_t = w[:12] @ r_ETF.loc[t]


        # Save to w_pd
        w.append(mu_0)
        w.append(sigma)
        w.append(cvar)
        w.append(var)
        w.append(r_bt_t)
        w.append(targ_vol)
        w_pd_CVaR.loc[i, :] = w
        cvar_keep, w_keep, var_keep = cvar, w, var


    sigma_p_max = w[12]
    if sigma_p_max <= targ_vol:
        w_opt_CVaR = w

    else:
        cvar, w, var, _ = CVar(beta=CL, R=0, returns_mat=bs_r, target_vol=targ_vol, mu=mu, Sigma=Sigma,
                               ESG_lb=ESG_cons,
                               include_ESG_rating_target=True,
                               include_target_vol=True, strict = True)
        mu_0 = w @ mu
        sigma = np.sqrt(w @ Sigma @ w)

        # Return realization
        r_bt_t = w[:12] @ r_ETF.loc[t]

        # Save to w_pd
        w.append(mu_0)
        w.append(sigma)
        w.append(cvar)
        w.append(var)
        w.append(r_bt_t)
        w.append(targ_vol)
        w_pd_CVaR.loc[i, :] = w
        w_opt_CVaR = w

    #### Model 2: Classic Minimum Variance Optimization ####
    # Logic: Same as CVaR.  First find the sigma_ps on gridded mu. Compare the sigma_p at mu_max
    #        If it is less than target volatility, then optimal w should be equal to
    #        the one obtained at mu.max. If not, interpolate the mu and back out the sigma_p.
    mu = np.array(bs_r.mean())
    Sigma = np.array(bs_r.cov())


    cname = list(r_ETF.columns)
    cname.append("mu")
    cname.append("sigma_p")  # portfolio sigma
    cname.append("Realized_r_t")
    cname.append("Target_vol")
    w_pd_MV = pd.DataFrame(columns=cname)


    sigma_list = []
    for i in mu_space:
        w = MinVar(i, mu, Sigma)
        if w is None:
            w = w_keep
            break
        sigma = float(np.sqrt(w.T @ Sigma @ w))
        sigma_list.append(sigma)
        w_keep = w


    if sigma_list[-1] <= targ_vol:
        w_opt_MV = w.T.tolist()[0]

    if sigma_list[-1] > targ_vol:
        mu_interpolate = np.interp(target_vol_bl, xp=sigma_list, fp = mu_space[:len(sigma_list)])
        w = MinVar(mu_interpolate, mu, Sigma)
        w_opt_MV = w.T.tolist()[0]

    mu_0 = w_opt_MV @ mu
    sigma = np.sqrt(w_opt_MV @ Sigma @ w_opt_MV)

    # Return realization
    r_bt_t = w_opt_MV[:12] @ r_ETF.loc[t]

    # Save to w_pd
    w_opt_MV.append(mu_0)
    w_opt_MV.append(sigma)
    w_opt_MV.append(r_bt_t)
    w_opt_MV.append(targ_vol)
    w_pd_MV.loc[i, :] = w_opt_MV

    # TODO: Information ratio
    return w_pd_CVaR, w_opt_CVaR, w_pd_MV, w_opt_MV


def train_test_split(df, cutting_points):
    """
    :param df: The df to be split
    :param cutting_points: The point to split the time series. A list of datetime
    :return: Dictionary of dataframe. The key is the staring date for testing.
    """
    dic = {}

    cutting_points0 = copy.copy(cutting_points)
    # Sort date
    cutting_points0.sort()

    # Add the beginning and end t
    cutting_points0.insert(len(cutting_points0), df.index[-1])
    cutting_points0.insert(0, df.index[0])

    for t in range(len(cutting_points0) - 1):
        df_test = df[cutting_points0[t]:cutting_points0[t + 1]]
        df_train = df[~np.isin(df.index, df_test.index)]
        df_sorted = pd.concat([df_train, df_test])

        # location of the first test date
        _1st_date_test = df_test.index[0]
        _1st_date_test_loc = list(df_sorted.index).index(_1st_date_test)

        # Re-index df_sorted
        df_sorted_index = df_sorted.index.sort_values()
        df_sorted.index = df_sorted_index

        # Relocate the starting test date
        _1st_date_test = df_sorted.index[_1st_date_test_loc]

        dic[_1st_date_test] = df_sorted

    return dic


def cv_once(r, g, pi, test_start,
            targ_vol_list = [0.15, 0.12, 0.1, 0.08, 0.05] / np.sqrt(12),
            mu_grid = 10):
    """
    :param: test_start: the 1st date of testing
    :return:
    """

    # Entire period
    _0T = r.index
    loc = list(_0T.values).index(test_start)

    pd_opt_cvar = pd.DataFrame()
    pd_opt_mv = pd.DataFrame()
    for t in _0T[loc:]:
        print(t)
        for tv in targ_vol_list:
            pd_cvar, w_opt_cvar, pd_mv, w_opt_mv = Walk_forward_CVaR(t, g, pi, r, N_mu_grid=mu_grid,
                                                                                 targ_vol=tv, bs_method="parametric")
            w_opt_cvar = pd.DataFrame(w_opt_cvar).T
            w_opt_cvar.columns = pd_cvar.columns
            pd_opt_cvar = pd.concat([pd_opt_cvar, pd.DataFrame(w_opt_cvar)])

            w_opt_mv = pd.DataFrame(w_opt_mv).T
            w_opt_mv.columns = pd_mv.columns
            pd_opt_mv = pd.concat([pd_opt_mv, pd.DataFrame(w_opt_mv)])

    return pd_opt_cvar, pd_opt_mv