import numpy as np
import pandas as pd
import prediction
from tqdm import tqdm
from IPython.display import clear_output
import sys

#------------------------------------------------------------------------------------

def rolling_window(df, window_size=30, horizon=7):
    for i in range(len(df)-horizon-window_size):
        df_x = df.iloc[i:i+window_size,:]

        df_y = df.iloc[i+window_size: i+window_size+horizon]

        progress = round(100* (i / (len(df)-horizon-window_size)), 2)

        yield(df_x, df_y, progress)

#----------------------------------------------------------------------------------

def rolling_evaluate(df, model, metric='smape', window_size=30, horizon=10, plot_examples=0, mean_stocks=False):

    x = rolling_window(df, window_size=window_size, horizon=horizon)

    windows_losses = []
    
    threshold = 100 / (plot_examples+1)

    for i in x:
        y_true = pd.Series(list(i[1].values.T), index=df.columns)

        if i[2] > threshold:
            y_pred = prediction.predict(i[0], model=model, horizon=horizon, plot=True)
            threshold += 100 / (plot_examples+1)
        else:
            y_pred = prediction.predict(i[0], model=model, horizon=horizon, plot=False)

        window_loss = smape(y_true=y_true, y_pred=y_pred)

        windows_losses.append(window_loss.to_list())

    windows_losses = np.mean(windows_losses, axis=0)
    windows_losses = pd.Series(windows_losses, index=df.columns)

    if mean_stocks:
        windows_losses = np.mean(windows_losses.to_list())

    return windows_losses    

#----------------------------------------------------------------------------------

def generate_benchmark(df, window_size=30, horizon=10):
    supported_models = ['naive', 'monte_carlo', 'moving_average', 'single_exponential_smoothing']
    columns = [f'Horizon {i+1}' for i in range(horizon)]

    table = pd.DataFrame(index = supported_models, columns=columns)

    for index, m in enumerate(supported_models):
        print(f'In Progress. Model {index+1} of {len(supported_models)}')
        for h in range(horizon):
            loss_i_j = rolling_evaluate(df, model=m, metric='smape', window_size=window_size, horizon=h+1, plot_examples=0, mean_stocks=True)

            table.iloc[index, h] = loss_i_j


    return table

#----------------------------------------------------------------------------------

def smape(y_true, y_pred):

    smapes = []
    for i in range(len(y_true)):
        denominator = (np.abs(y_true[i]) + np.abs(y_pred[i])) / 2.0
        diff = np.abs(y_true[i] - y_pred[i]) / denominator
        diff[denominator == 0] = 0
        smape_value = np.mean(diff) * 100

        smapes.append(smape_value)

    smapes = pd.Series(smapes, index=y_true.index)

    return smapes



