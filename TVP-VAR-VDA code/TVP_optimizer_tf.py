import numpy as np
import tensorflow_probability as tfp
import tensorflow.compat.v2 as tf


class TVP_variational_optimization:

    def __init__(self, n_dim, K, window, Q_0, beta_b, t, Y, time):
        # n_dim x n_dim
        sigma_bar = np.identity(n_dim) * 0.05

        self.sigma_inv = tf.convert_to_tensor(tf.linalg.inv(sigma_bar), dtype=tf.float64)
        self.Q_inv = tf.convert_to_tensor(tf.linalg.inv(Q_0), dtype=tf.float64)
        # background beta
        self.beta_b = tf.convert_to_tensor(beta_b, dtype=tf.float64)

        self.beta = tf.Variable(np.zeros((n_dim * K, 1)), dtype=tf.float64)
        self.n_dim = n_dim
        self.time = time
        self.Y = tf.convert_to_tensor(Y, dtype=tf.float64)
        self.K = K
        self.t = t
        self.window = window

    def cost(self, x):
        self.beta = tf.reshape(x, (x.shape[0], 1))
        beta_b_beta = (self.beta_b - self.beta)
        beta_b_beta_transpose = tf.transpose(beta_b_beta)

        beta_b_beta_Q_0 = tf.matmul(beta_b_beta_transpose, self.Q_inv)
        J_b = tf.matmul(beta_b_beta_Q_0, beta_b_beta)
        J_0 = 0

        if self.window == 1:
            Y_new = tf.reshape(self.Y[self.t], [self.n_dim, 1])
            residual = Y_new - tf.matmul(self.X[0], self.beta)
            residual_t = tf.transpose(residual)
            residula_sigma = tf.matmul(residual_t, self.sigma_inv)
            J_0 = tf.matmul(residula_sigma, residual)
        else:
            effective_window = self.window
            if self.t + effective_window > self.time:
                effective_window = self.time - self.t
            t = self.t
            tau = self.t + effective_window

            x_beta = tf.einsum("abc,cd->abd", self.X, self.beta)
            Y_new = tf.reshape(self.Y[t:tau], [effective_window, self.n_dim, 1])
            residual = Y_new - x_beta
            residual_t = tf.transpose(residual, perm=[0, 2, 1])
            residual_sigma = tf.einsum("abc,cd->abd", residual_t, self.sigma_inv)
            total_residual_cost = tf.einsum("abc,acd->abd", residual_sigma, residual)
            J_0 = tf.reduce_sum(total_residual_cost, 0)
        return tf.reshape(J_b + J_0, ())

    def loss_and_gradient(self, x):
        return tfp.math.value_and_gradient(self.cost, x)

    def optimize_w(self):
        start = tf.reshape(self.beta_b, (self.beta_b.shape[0]))
        output = tfp.optimizer.lbfgs_minimize(lambda x: tfp.math.value_and_gradient(self.cost, x),
                                              initial_position=start, tolerance=1e-8)
        return output.position
