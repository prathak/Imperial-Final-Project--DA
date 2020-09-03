import numpy as np
import time as timee
import tensorflow as tf



def run_DGP(periods, dimensions, lag=False):
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

    # For lagg 2
    if (lag):
        for i in np.arange(0, n_dim):
            for j in np.arange(2, t).reshape(-1):
                Beta[j][i] = Beta[j - 1][i] + e2[j][i]
                Y[j][i] = X[j][i] * Beta[j][i] + X[j - 1][i] * Beta[j - 1][i] + X[j - 2][i] * Beta[j - 2][i] + e1[j][
                    i] + e3[j][i]  # +e4[j][n_dim-1]
    else:
        for i in np.arange(0, n_dim):
            for j in np.arange(1, t).reshape(-1):
                Beta[j][i] = Beta[j - 1][i] + e2[j][i]
                Y[j][i] = X[j][i] * Beta[j][i] + e1[j][i] + e3[j][i]  # +e4[j][n_dim-1]

    X_raw = X
    Y_raw = Y

    return X_raw, Y_raw, Beta


def get_priors(n_dim, K):
    # n_dim x n_dim
    sigma_bar = tf.eye(n_dim, dtype=tf.float64) * 0.05

    # Random prior initialization. Not sure if this need to be well informed initialisation or random.
    Q_0 = tf.eye(n_dim * K, dtype=tf.float64) * 1.01
    beta_0 = np.eye(n_dim*K, 1)*0.06
    beta_tensor = tf.convert_to_tensor(beta_0, dtype=tf.float64)
    return sigma_bar, Q_0, beta_tensor


# creating x = I ⊗ X, Xt = [y t−1 0 ,...,y t−l ,1]
def get_x(time_dim, X):
    X = np.concatenate((X, np.ones(shape=[time_dim, 1])), axis=1)
    return X


def tvp_da(sigma_bar, Q_0, beta_0, X, Y, time, K, n_dim, lag_length):
    qK = n_dim * K
    beta = tf.unstack(tf.zeros([time, qK, 1]))
    beta[lag_length] = beta_0
    Q_prev = Q_0
    Y_hat = tf.unstack(tf.zeros(shape=[time, n_dim]))
    avg = 0
    I_2 = tf.eye(qK, dtype=tf.float64)
    I = tf.eye((K * n_dim), dtype=tf.float64)
    for t in range(lag_length + 1, time):
        start1 = timee.time()
        x = tf.convert_to_tensor(np.kron(np.identity(n_dim), X[t]), dtype=tf.float64)
        xT = tf.transpose(x)
        x_Q_xT = tf.matmul(tf.matmul(x, Q_prev), xT)
        inv = tf.linalg.inv(x_Q_xT + sigma_bar)
        kalman_gain = tf.matmul(tf.matmul(Q_prev, xT), inv)
        y_tensor = tf.convert_to_tensor(Y[t], dtype=tf.float64)
        Y_new = tf.reshape(y_tensor, (n_dim, 1))
        y_hat = tf.matmul(x, beta[t - 1])
        Y_hat[t] = np.reshape(y_hat, n_dim)
        beta[t] = beta[t - 1] + tf.matmul(kalman_gain, tf.subtract(Y_new, y_hat))
        Q = (I_2 - tf.matmul(tf.matmul(kalman_gain, x), Q_prev))
        Q_prev = Q + I
        start2 = timee.time()
        avg += start2 - start1
    print("Avg time : ", avg / time)
    return tf.stack(beta), tf.stack(Y_hat)


def mse_error(y_hat, y):
    return ((y.T - y_hat.T) ** 2).mean(axis=1).mean()


def run(time_dim, n_dim):
    lag = False
    for lag_length in [0]:
        K = n_dim + 1 if lag_length == 0 else (lag_length + 1) * n_dim + 1
        print("Lag : " + str(lag_length))
        X, Y, _ = run_DGP(periods=time_dim, dimensions=n_dim, lag=lag)
        sigma_bar, Q_0, beta_0 = get_priors(n_dim, K)
        print("Done with data")
        start = timee.time()
        x = get_x(time_dim, X)
        print("In tvp da algo")
        beta, y_hat = tvp_da(sigma_bar, Q_0, beta_0, x, Y, time_dim, K, n_dim, lag_length)
        end = timee.time()
        print('runtime :' + str(lag_length) + ' :', end - start)
        print('time :' + str(time_dim) + ' and n_dim :' + str(n_dim))
        lag = True
        print("Mse error :" + str(mse_error(y_hat.numpy(), Y)))


