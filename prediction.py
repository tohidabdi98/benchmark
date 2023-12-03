import numpy as np
import pandas as pd
import plots
import statistic
from scipy.stats import t
import load
import statsmodels.api as sm

#-------------------------------------------------------------------------------------

def naive_prediction(df, horizon=10, plot=False):
    train = df.iloc[:-horizon,:]
    test = df.iloc[-horizon:,:]

    predictions = []
    for i in range(len(train.columns)):
        price_list = np.full(horizon, train.iloc[-1,i])
    
        if plot:
            plots.plot_series(train.iloc[:,i], test.iloc[:,i], price_list, stock=train.columns[i])


        predictions.append(price_list)


    predictions = pd.Series(predictions, index=train.columns)

    return predictions

#-------------------------------------------------------------------------------------

def monte_carlo_simulation(data, horizon=10, iterations=4, central_measure='mean', plot=False, return_central=False, seed=4):
    train = data.iloc[:-horizon,:]
    test = data.iloc[-horizon:,:]

    np.random.seed(seed)
    means = statistic.calculate_returns(train, log_returns=True, annual=False)

    predictions = []
    for i in range(len(train.columns)):

        mean = means.iloc[:,i].mean()
        var = means.iloc[:,i].var()
        std = means.iloc[:,i].std()

        drift = mean - (0.5 * var)

        daily_returns = np.exp(drift + std * t.ppf(np.random.rand(horizon+1, iterations), df=5))

        price_list = np.zeros_like(daily_returns)
        price_list[0] = train.iloc[-1,i]

        for h in range(1, horizon+1):
            price_list[h] = price_list[h-1] * daily_returns[h]

        price_list = price_list[1:]


        if iterations < 2:
            price_list = price_list.flatten()

        if return_central:
            if price_list.ndim == 1:
                price_list = np.array(price_list)
            else:
                if central_measure == 'mean':
                    price_list = np.mean(price_list, axis=1)
                elif central_measure == 'median':
                    price_list = np.median(price_list, axis=1)
                price_list = price_list.flatten()

        if plot:
            plots.plot_series(train.iloc[:,i], test.iloc[:,i], price_list, stock=train.columns[i])

        predictions.append(price_list)
    
    predictions = pd.Series(predictions, index=train.columns)

    return predictions

#-----------------------------------------------------------------------------------

def moving_average_predict(df, horizon=10, window=8, plot=False):
    train = df.iloc[:-horizon, :]
    test = df.iloc[-horizon:, :]

    predictions = []
    for i in range(len(train.columns)):
        pred = []
        serie = train.iloc[:, i]

        for j in range(horizon):

            ma = serie.iloc[-window:].mean()
            pred.append(ma)

            ma = pd.Series(ma, index=[test.index[j]])

            serie = pd.concat([serie, ma])

        if plot:
            plots.plot_series(train.iloc[:,i], test.iloc[:,i], np.array(pred), stock=train.columns[i])

        predictions.append(pred)

    predictions = pd.Series(predictions, index=train.columns)

    return predictions

#-----------------------------------------------------------------------------------

def single_exponential_smoothing(df, horizon=10, plot=False):
    train = df.iloc[:-horizon, :]
    test = df.iloc[-horizon:, :]

    predictions = []
    for i in range(len(train.columns)):

        ses_model = sm.tsa.SimpleExpSmoothing(train.iloc[:,i], initialization_method="estimated").fit()
        ses = ses_model.forecast(horizon)

        if plot:
            plots.plot_series(train.iloc[:,i], test.iloc[:,i], np.array(ses), stock=train.columns[i])

        predictions.append(np.array(ses))

    predictions = pd.Series(predictions, index=train.columns)

    return predictions
 

#-----------------------------------------------------------------------------------

def predict(df, model, horizon=10, seed=4, plot=False):
    if model == 'monte_carlo':
        predictions = monte_carlo_simulation(df, horizon=horizon, iterations=1, plot=plot, return_central=False, seed=4)

    elif model == 'naive':
        predictions = naive_prediction(df, horizon=horizon, plot=plot)

    elif model == 'moving_average':
        predictions = moving_average_predict(df, horizon=horizon, window=30, plot=plot)

    elif model == 'single_exponential_smoothing':
        predictions = single_exponential_smoothing(df, horizon=horizon, plot=plot)
  
    return predictions