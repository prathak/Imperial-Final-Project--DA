import numpy as np
from gen_data import run_DGP
import tvp_var_vda
import plot as plot_beta


def run_tvp_var(n_dim, time_dim, window):
    K, Q_0, beta, beta_b, qK, tt, I_torch = get_params(n_dim, time_dim)

    X, Y, Beta = run_DGP(periods=time_dim, dimensions=n_dim)
    X = np.concatenate((X, np.ones(shape=[time_dim, 1])), axis=1)

    beta, Y_hat = tvp_var_vda.run_tvp_var(X, Y, time_dim, n_dim, K, Q_0, beta, beta_b, qK, I_torch, window)
    plot_beta.plot(time_dim, n_dim, beta, Beta, Y_hat, Y)


def get_params(n_dim, time_dim):
    tt = 0
    K = n_dim + 1
    Q_0 = (np.eye((n_dim * K)) * 1.01)
    beta_b = np.eye(n_dim * K, 1) * 0.06
    qK = n_dim * K
    beta = np.zeros([time_dim, qK, 1])
    I_torch = np.eye(n_dim)

    return K, Q_0, beta, beta_b, qK, tt, I_torch


run_tvp_var(2, 200, 1)