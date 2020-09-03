from gen_data import get_blockchain_data_with_datetime, get_blockchain_data
import numpy as np
import tvp_var_vda
import plot as plot_beta
import pandas as pd


def run_tvp_var(off_chain_file_name, on_chain_file_name):
    X, Y, datetime_list = get_blockchain_data_with_datetime(off_chain_file_name=off_chain_file_name,
                                                            on_chain_file_name=on_chain_file_name,swap=False)

    time_dim, n_dim, K, Q_0, beta, beta_b, qK, I_torch, window = get_params(X, Y)
    beta, Y_hat = tvp_var_vda.run_tvp_var(X, Y, time_dim, n_dim, K, Q_0, beta, beta_b, qK, I_torch, window)
    beta_pd = pd.DataFrame(beta[:, :, 0])
    beta_pd.index = datetime_list
    index_y = 2
    index_x = 2
    num = X.shape[1]
    pos = index_y * num + index_x
    plot_beta.plot_beta_dataframe_with_datetime(beta_pd, pos, 'BNB_ammount_to_price_diff_volume_diff_Binance_beta_swap',
                                                True)


def get_params(X, Y):
    time_dim = X.shape[0]
    n_dim = (Y.shape[1])
    window = 1
    K = X.shape[1]
    # K = n_dim
    Q_0 = np.eye(n_dim * K) * 2.05
    beta_b = np.eye(n_dim * K, 1) * 0.06
    qK = X.shape[1] * n_dim
    beta = np.zeros([time_dim, qK, 1])
    I_torch = np.eye(n_dim)

    return time_dim, n_dim, K, Q_0, beta, beta_b, qK, I_torch, window


file_name_off_chain = "./blockchain_data/BNB_ammount_to_price_diff_volume_diff.csv"
file_name_on_chain = './blockchain_data/BNB_amount_to_data_volume_diff.csv'
run_tvp_var(off_chain_file_name=file_name_off_chain, on_chain_file_name=file_name_on_chain)
