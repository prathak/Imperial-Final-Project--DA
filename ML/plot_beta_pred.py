import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter


def plot_beta_with_datetime(beta_dataframe, pred_df, pred_test, index, tot, pos, filename, save=False):

    y = mdates.datestr2num(beta_dataframe.index.values)

    sns.set(font_scale=1.5, style="white")

    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.xaxis_date()
    ax.plot(y[:-20],
            beta_dataframe.iloc[:-20, pos], label="true")
    ax.plot(y[20:index],
            pred_df.iloc[:, pos], linestyle="-.", label="train-predicted", color="burlywood")
    ax.plot(y[index:tot - 20],
            pred_test.iloc[:, pos], linestyle="-.", label="test-predicted", color="yellowgreen")

    ax.set(xlabel="Date",
           ylabel="Latent Parameter beta")
    date_form = DateFormatter("%y-%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    plt.legend()
    if save:
        plt.savefig(filename)

    plt.show()
