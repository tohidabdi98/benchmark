import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------------

def get_tickers_list(index):
    if index == 'S&P500':
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        data = pd.read_html(url)
        table = data[0]
        tickers = table['Symbol'].tolist()
        tickers = [ticker.replace('BF.B', 'BF-B').replace('BRK.B', 'BRK-B') for ticker in tickers]
        tickers = sorted(tickers)
        tickers.insert(0, '^GSPC')

    elif index == 'dow_average':
        url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        data = pd.read_html(url)
        table = data[1]
        tickers = table['Symbol'].tolist()
        tickers = sorted(tickers)
        tickers.insert(0, '^DJI')

    elif index == 'dax':
        url = 'https://en.wikipedia.org/wiki/DAX'
        data = pd.read_html(url)
        table = data[4]
        tickers = table['Ticker'].tolist()
        tickers = sorted(tickers)
        tickers.insert(0, '^GDAXI')
    
    tickers = np.array(tickers)

    return tickers

#--------------------------------------------------------------------------

def download_data(index, type='c'):
    data = pd.DataFrame()
    tickers = get_tickers_list(index)

    print('Downloading')

    if type == 'c':
        for t in tqdm(tickers):
            ticker = yf.Ticker(t)
            data[t] = ticker.history(start='1800-01-01')['Close']

    elif type == 'ohlcv':
        pass

    return data

#---------------------------------------------------------------------------

def normalize_date(df):
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.strftime('%Y/%m/%d')
    df.index = pd.to_datetime(df.index, utc=True)

    return df

#----------------------------------------------------------------------------

def normalize_start_date(dataframe, plot=True):
    dataframe.fillna(method='ffill', inplace=True)
    years = np.arange(1800, 2024, 1, dtype=int)
    elements = []

    for year in years:
        trimmed_data = dataframe.loc[str(str(year)+'-01-01'):]
        trimmed_data = trimmed_data.dropna(axis=1)
        elements.append(trimmed_data.size)

    if plot:
        plt.plot(years, elements)
        plt.show()

    opt_data = dataframe.loc[str(years[np.argmax(elements)])+'-01-01':]
    opt_data = opt_data.dropna(axis=1)

    return opt_data

#-----------------------------------------------------------------------------

def set_frequency(df, frequency='b'):
    df_freq = df.asfreq('b', normalize=True)
    df_freq_filled = df_freq.interpolate()

    return df_freq_filled

#-----------------------------------------------------------------------------

def load_df(index, type='c', optimize_start_date_plot=True):
    if os.path.exists(f'dataframes/{index}_{type}.csv'):
        df = pd.read_csv(f'dataframes/{index}_{type}.csv', index_col='Date')
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.asfreq('b')
        print('Dataframe Loaded')

    else:
        data = download_data(index)
        normalized_data = normalize_date(data)

        start_normalized_data = normalize_start_date(normalized_data, plot=optimize_start_date_plot)

        freq_data = set_frequency(start_normalized_data)

        if not os.path.exists('dataframes'):
            os.mkdir('dataframes')

        freq_data.to_csv(f'dataframes/{index}_{type}.csv')
        df = freq_data.asfreq('b')

    return df

#----------------------------------------------------------------------------

def train_validation_test_split(df, validation_split=True, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if train_ratio + val_ratio + test_ratio != 1.0:
        print('Ratios are not equal to 1')
    
    else:
        train_validation, test = train_test_split(df, test_size=test_ratio, shuffle=False)

        if validation_split:
            train, validation = train_test_split(train_validation, test_size=(val_ratio/(train_ratio+val_ratio)), shuffle=False)

            return train, validation, test
        
        else:
            return train_validation, test
        
#------------------------------------------------------------------------------