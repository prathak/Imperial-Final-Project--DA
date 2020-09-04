import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import seaborn as sns

def get_transformed_beta(time, n_dim, beta):
    transformed = np.zeros((time, n_dim), dtype=np.float)
    for t in range(0, time):
        a = int(beta[t, :, 0].shape[0] / n_dim)
        b2 = np.reshape(beta[t, :, 0], (n_dim, a))
        transformed[t] = np.diag(b2)
    return transformed

def plot(time_dim, n_dim, beta, Beta, y_hat, Y):
    transformed_beta = get_transformed_beta(time_dim, n_dim, beta)
    fig2, ax = plt.subplots(1, 2, figsize=(17, 5))
    ax[0].set_title(r'Beta')
    ax[0].plot(transformed_beta[:, :], linestyle='--', label="Predicted")
    ax[0].plot(Beta[:, :], label="True")
    ax[0].legend()
    ax[1].set_title(r'Real and Forecasted values $Y$')
    ax[1].plot(y_hat, linestyle='--', label="Predicted")
    ax[1].plot(Y, label="True")
    ax[1].legend()
    plt.pause(0.005)
    plt.tight_layout()

    plt.savefig('temp_synthetic2.png')
    plt.show()

def plot_beta_dataframe_with_datetime(beta_dataframe, pos, filename, save=False):
    y = mdates.datestr2num(beta_dataframe.index.values)

    sns.set(font_scale=1.5, style="white")

    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.xaxis_date()
    ax.plot(y,
            beta_dataframe.iloc[:, pos],
            color='purple')

    ax.set(xlabel="Date",
           ylabel="Latent Parameter beta",
           title="Date")
    date_form = DateFormatter("%y-%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    if save:
        plt.savefig(filename + '.png')
    plt.show()
