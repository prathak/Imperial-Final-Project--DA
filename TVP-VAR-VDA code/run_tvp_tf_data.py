import argparse

from gen_data import get_blockchain_data_with_datetime, get_blockchain_data
import numpy as np
import tvp_var_vda
import plot as plot_beta
import pandas as pd


def run_tvp_var(off_chain_file_name, on_chain_file_name, save_location, index_x, index_y):
    X, Y, datetime_list = get_blockchain_data_with_datetime(off_chain_file_name=off_chain_file_name,
                                                            on_chain_file_name=on_chain_file_name, swap=False)

    time_dim, n_dim, K, Q_0, beta, beta_b, qK, I_torch, window = get_params(X, Y)
    beta, Y_hat = tvp_var_vda.run_tvp_var(X, Y, time_dim, n_dim, K, Q_0, beta, beta_b, qK, I_torch, window)
    beta_pd = pd.DataFrame(beta[:, :, 0])
    beta_pd.index = datetime_list
    num = X.shape[1]
    pos = index_y * num + index_x
    plot_beta.plot_beta_dataframe_with_datetime(beta_pd, pos, save_location,
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


def get_arguments():
    parser = argparse.ArgumentParser(description='LSTNet Model')

    parser.add_argument('--on_chain_file', type=str, required=True, help='On chain file location')
    parser.add_argument('--off_chain_file', type=str, required=True, help='Off chain file location')
    parser.add_argument('--save', type=str, required=True, help='Plot save location and file name')
    parser.add_argument('--index_x', type=int, required=True, help='Index of exchange for dataset used for on_chain')
    parser.add_argument('--index_y', type=int, required=True, help='Index of exchange for dataset used for off_chain')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    try:
        args = get_arguments()
    except SystemExit as err:
        print("Error reading arguments")
        exit(0)

    run_tvp_var(off_chain_file_name=args.off_chain_file, on_chain_file_name=args.on_chain_file,
                save_location=args.save, index_x=args.index_x, index_y=args.index_y)

# file_name_off_chain = "./blockchain_data/BNB_ammount_to_price_diff_volume_diff.csv"
# file_name_on_chain = './blockchain_data/BNB_amount_to_data_volume_diff.csv'
# run_tvp_var(off_chain_file_name=file_name_off_chain, on_chain_file_name=file_name_on_chain)
