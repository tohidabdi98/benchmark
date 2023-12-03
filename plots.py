import datetime
import matplotlib.pyplot as plt
import pandas as pd

#----------------------------------------------------------------------------------


def plot_series(x, y_true, y_pred, stock):
    plt.figure(figsize=(20,10))

    plt.plot(x, label='Actual Data')

    plt.axvline(x=x.index[-1], color='red', linestyle='--', label='Threshold on Last Day of X')

    if y_pred.ndim > 1:
        for i in range(len(y_pred.T)):
            y_pred_i = pd.Series(y_pred.T[i], index=y_true.index)

            plt.plot(y_pred_i, label=f'Predicted Values_{i}')

    else:
        y_pred = pd.Series(y_pred, index=y_true.index)

        plt.plot(y_pred, label='Predicted Values')

    # print(y_pred)
    # print(y_true)

    plt.plot(y_true, label='True Values')

    plt.legend()

    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(f'Predictions of Model for {stock}')

    # Show the plot
    plt.show()

#-------------------------------------------------------------------------------------