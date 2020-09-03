import tensorflow as tf
from TVP_optimizer_tf import TVP_variational_optimization
import numpy as np
import tensorflow.compat.v2 as tf
import time


def mse_error(y_hat, y):
    sq_error = (tf.transpose(y) - tf.transpose(y_hat)) ** 2
    return tf.math.reduce_mean(sq_error, 1)

def run_tvp_var(X, Y, time_dim, n_dim, K, Q_0, beta, beta_b, qK, I, window):
    x = np.zeros((window, n_dim, qK))

    tvp_var_optimizer = TVP_variational_optimization(n_dim, K, window, Q_0, beta_b, 0, Y, time_dim)
    Y_hat = np.zeros(shape=[time_dim, n_dim])
    start = time.time()
    for t in range(0, time_dim, window):
        if t % 100 == 0:
            print("At time : " + str(t))

        for tau in range(0, window):
            if t + tau >= time_dim:
                break
            x[tau] = np.kron(I, X[t + tau, :].T)

        tvp_var_optimizer.X = tf.convert_to_tensor(x, dtype=tf.float64)

        tvp_var_optimizer.t = t

        optim = tvp_var_optimizer.optimize_w().numpy()
        beta[t] = optim.reshape(optim.shape[0], 1)

        # Copying over beta{t} ot beta{t+1}..beta{t + tau} as Forecaset matrix F = Identity.
        for tau in range(0, window):
            if t + tau >= time_dim:
                break
            beta[t + tau] = beta[t]
            Y_hat[t + tau] = np.reshape((x[tau] @ beta[t]), n_dim)
        # background beta update
        beta_b = beta[t]
        tvp_var_optimizer.beta_b = tf.convert_to_tensor(beta_b, dtype=tf.float64)
    end = time.time()
    print("For n_dim : " + str(n_dim) + " and time : ", str(time_dim) + " and window : " + str(window))
    print('runtime :', end - start)
    print('runtime mins:', (end - start) / 60)
    print("Mse error :" + str(mse_error(Y_hat, Y)))
    return beta, Y_hat
