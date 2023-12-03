import numpy as np
import pandas as pd

#----------------------------------------------------------

def calculate_returns(df, log_returns=False, periods=1, annual=False, days_per_year=252):
    if log_returns:
        if annual:
            returns = np.log(1 + df.pct_change(periods)).mean() * days_per_year
        else:
            returns = np.log(1 + df.pct_change(periods))
    else:
        if annual:
            returns = df.pct_change(periods=periods).mean() * days_per_year
        else:
            returns = df.pct_change(periods=periods)

    if not annual:
        returns = returns.iloc[1:,:]

    return returns

#----------------------------------------------------------------------------------

def calculate_stds(df, annual=False, days_per_year=252):
    if annual:
        stds = calculate_returns(df, annual=False, days_per_year=days_per_year).std() * days_per_year ** 0.5
    else:
        stds = calculate_returns(df, annual=False, days_per_year=days_per_year).std()

    return stds

#--------------------------------------------------------------------------------

def calculate_vars(df, annual=False, days_per_year=252):
    if annual:
        vars = calculate_returns(df, annual=False, days_per_year=days_per_year).var() * days_per_year
    else:
        vars = calculate_returns(df, annual=False, days_per_year=days_per_year).var()

    return vars

#---------------------------------------------------------------------------------

def returns_cov(df, annual=False, days_per_year=252):
    if annual:
        cov = calculate_returns(df, annual=False, days_per_year=days_per_year).cov() * days_per_year
    else:
        cov = calculate_returns(df, annual=False, days_per_year=days_per_year).cov()

    return cov

#----------------------------------------------------------------------------------

def returns_correlation(df):
    corr = calculate_returns(df, annual=False).corr()

    return corr

#----------------------------------------------------------------------------------

def calculate_betas(df, index=0):
    betas = []
    for i in range(len(df.columns)):
        cov_with_market = returns_cov(df, annual=True).iloc[0,i]
        beta = cov_with_market / calculate_vars(df, annual=True)[index]
        betas.append(beta)

    series = pd.Series(betas, index=df.columns)

    return series

#-----------------------------------------------------------------------------------

def capm_expected_returns(df, risk_free_rate=0.0):
    capm_expected_returns = risk_free_rate + (calculate_betas(df) * (calculate_returns(df, annual=True) - risk_free_rate))

    return capm_expected_returns

#----------------------------------------------------------------------------------

def sharpe_ratio(df, risk_free_rate=0.0):
    sharpe_ratios = (calculate_returns(df, annual=True)-risk_free_rate) / calculate_stds(df, annual=True)

    return sharpe_ratios

#----------------------------------------------------------------------------------