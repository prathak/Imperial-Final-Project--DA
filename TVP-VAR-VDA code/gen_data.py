import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def run_DGP(periods, dimensions):
    ##Data Generating Process######
    np.random.seed(1337)  # this is the only part of the whole code which is stochastic

    n_dim = dimensions  # INPUT # number of dimensions #

    t = np.array(periods)  # INPUT #500

    # try out different scales
    Q = np.array(0.001)  # 0.001
    R = np.array(0.01)  # 0.01
    Q_test = 0.001
    R_test = 0.01

    # generate white noise series
    e1 = np.random.normal(0, 1, [t, n_dim])
    # e1 = e1*Q
    e1 = e1 * Q

    e2 = np.random.normal(0, 1, [t, n_dim])
    e2 = e2 * R

    e3 = np.random.normal(0, 1, [t, n_dim])
    e3 = e3 * Q_test

    # generate containers
    Beta = np.zeros(shape=[t, n_dim])
    Y = np.zeros(shape=[t, n_dim])

    # generate  X as N-(3,1) process
    X = np.random.normal(3, 1, size=[t, n_dim])

    # generate state variable beta as random walk process, and Y as a function of this.

    for i in np.arange(0, n_dim):
        for j in np.arange(1, t).reshape(-1):
            Beta[j][i] = Beta[j - 1][i] + e2[j][i]
            Y[j][i] = X[j][i] * Beta[j][i] + e1[j][i] + e3[j][i]  # +e4[j][n_dim-1]

    return X, Y, Beta


def get_blockchain_data_with_datetime(off_chain_file_name, on_chain_file_name,swap = False):
    off_chain = pd.read_csv(off_chain_file_name, index_col=[0])

    volumeto_col = [col for col in off_chain if col.startswith('volumeto')]
    volumefrom_col = [col for col in off_chain if col.startswith('volumefrom')]
    volume_chain = off_chain[volumefrom_col]
    # df_on_chain = volume_chain.copy()

    df_off_chain = off_chain.drop(volumeto_col + volumefrom_col + ['datetime'], axis=1)

    df_on_chain = pd.read_csv(on_chain_file_name, index_col=[0])
    # df_on_chain = df_on_chain[(df_on_chain.T != 0).any()]

    df_on_chain = df_on_chain[df_on_chain.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

    sc_in = MinMaxScaler(feature_range=(0, 1))
    scaled_columns = df_on_chain.reset_index().drop('ts', axis=1).columns
    scaled_input = sc_in.fit_transform(df_on_chain[scaled_columns])
    df_on_chain[scaled_columns] = scaled_input
    df_on_chain[df_on_chain[scaled_columns] > 0]+=1

    df_off_chain = df_off_chain[df_off_chain.index.isin(df_on_chain.index)]

    y_raw = df_off_chain.to_numpy()[1:]
    x_raw = df_on_chain.to_numpy()[1:]

    time_dim = x_raw.shape[0]

    # Comment this to not swap inputs
    if swap:
        x_raw, y_raw = y_raw, x_raw
    x_raw = np.concatenate((x_raw, np.ones(shape=[time_dim, 1]) * 0.03), axis=1)

    datetime_list = off_chain[off_chain.index.isin(df_on_chain.index)]['datetime'].tolist()
    print("On chain col seq : ", df_on_chain.columns)
    print("Off chain col seq : ", df_off_chain.columns)
    return x_raw, y_raw, datetime_list[1:]


def get_blockchain_data(off_chain_file_name, on_chain_file_name,swap=False):
    df_off_chain = pd.read_csv(off_chain_file_name, index_col=[0])

    df_on_chain = pd.read_csv(on_chain_file_name, index_col=[0])
    df_on_chain = df_on_chain[(df_on_chain.T != 0).any()]

    df_on_chain = df_on_chain[df_on_chain.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

    sc_in = MinMaxScaler(feature_range=(0, 1))
    scaled_columns = df_on_chain.reset_index().drop('ts', axis=1).columns
    scaled_input = sc_in.fit_transform(df_on_chain[scaled_columns])
    df_on_chain[scaled_columns] = scaled_input
    df_on_chain[df_on_chain[scaled_columns] > 0] += 1

    df_off_chain = df_off_chain[df_off_chain.index.isin(df_on_chain.index)]

    y_raw = df_off_chain.to_numpy()[1:]
    x_raw = df_on_chain.to_numpy()[1:]

    time_dim = x_raw.shape[0]

    #  Uncomment this to swap inputs
    if swap:
        x_raw, y_raw = y_raw, x_raw
    x_raw = np.concatenate((x_raw, np.ones(shape=[time_dim, 1]) * 0.03), axis=1)

    print("On chain col seq : ", df_on_chain.columns)
    print("Off chain col seq : ", df_off_chain.columns)
    return x_raw, y_raw, []
