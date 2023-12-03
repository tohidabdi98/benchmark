from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
import matplotlib.pyplot as plt

#---------------------------------------------------------

def seasonal_decomposition(df, model='multipicative'):
    for col in df.columns:
        s_dec = seasonal_decompose(df[col], model=model)
        s_dec.plot()
        plt.show()

#------------------------------------------------------------------------------

def autocorrelation_function(df, partial_autocorrelation, lags=40):
    for col in df.columns:
        if partial_autocorrelation:
            sgt.plot_pacf(df[col], lags=lags, zero=False)
            plt.title(f"PACF {col}")
            plt.show()
        else:
            sgt.plot_acf(df[col], lags=lags, zero=False)
            plt.title(f"ACF {col}")
            plt.show()

#-------------------------------------------------------------------------------

def augmented_dickey_fuller(df):
    for col in df.columns:
        print(col+':')
        print(sts.adfuller(df[col]))

#-------------------------------------------------------------------------------