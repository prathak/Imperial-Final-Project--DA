import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import TVP_VARNet_Model
import plot_beta_pred
from sklearn.metrics import median_absolute_error

sns.set(rc={'figure.figsize': (9, 8)})

np.random.seed(32)


def create_dataset(Betas, time_steps=2):
    Xs, ys = [], []
    for i in range(len(Betas) - time_steps):
        v = Betas[i:(i + time_steps)]
        # Uncomment this to use it to create input for network with only dense layer
        # Xs.append(np.reshape(v, time_steps*Betas.shape[1]))
        Xs.append(v)
        ys.append(Betas[i + time_steps])
    return np.array(Xs), np.array(ys)


def get_train_and_test_data(b_var, test_percentage=0.1, time_steps=10):
    tot = b_var.shape[0]
    test_pct = int(test_percentage * tot)

    b_var_train = b_var[:tot - test_pct]
    b_var_test = b_var[tot - test_pct:]
    X_train, Y_train = create_dataset(b_var_train, time_steps)
    X_test, Y_test = create_dataset(b_var_test, time_steps)

    return X_train, Y_train, X_test, Y_test


def get_scaled_b_var(data, scaler=None):
    scaler = MinMaxScaler((0, 1))
    scaled_b_var = scaler.fit_transform(data)
    return scaled_b_var, scaler


def get_unscaled_b_var(data, scaler):
    scaled_b_var = scaler.inverse_transform(data)
    return scaled_b_var, scaler


def get_beta(file, scaling_factor=1, numpy_type=False):
    beta_pd = pd.read_csv(file, index_col=[0])
    b_var = beta_pd.to_numpy()
    b_var = np.reshape(b_var, (b_var.shape[0], b_var.shape[1]))
    plt.plot(b_var[:,14])
    plt.savefig('beta.png')
    return b_var * scaling_factor


def run_model(file, scale, pos):
    # b_var,scaler = get_scaled_b_var(b_var)
    b_var = get_beta(file, scale)
    X_train, Y_train, X_test, Y_test = get_train_and_test_data(b_var, time_steps=20)

    input_shape = X_train.shape
    output_shape = Y_train[0].shape[0]
    print(input_shape, output_shape)
    model = TVP_VARNet_Model.get_TVP_VARNet_model(input_shape, output_shape)

    model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
    history = model.fit(X_train, Y_train, epochs=7, batch_size=30, verbose=1)
    plot_predictions(model, X_train, X_test, file, pos,scale)

    model.save('test.h5')


def plot_predictions(model, X_train, X_test, file, pos,scale):
    y_pred = get_predictions(model, X_train)
    y_test_pred = get_predictions(model, X_test)

    beta_df = pd.read_csv(file, index_col=[0]) * scale
    tot = beta_df.shape[0]
    test_pct = int(0.1 * tot)

    beta_pred = pd.DataFrame(y_pred)
    beta_pred_test = pd.DataFrame(y_test_pred)
    plot_beta_pred.plot_beta_with_datetime(beta_df, beta_pred, beta_pred_test, tot - test_pct, tot, pos, 'test.png',
                                           save=True)


def get_predictions(model, X):
    y_pred = model.predict(X)
    return y_pred


def get_error(y_pred, y_test_pred, Y_train, Y_test):
    print("My own train mae, mse : ", np.mean(np.mean(abs(y_pred - Y_train), axis=0)),
          np.mean(np.mean((y_pred - Y_train) ** 2, axis=0)))

    print("My own test mae, mse : ", np.sum(np.mean(abs(y_test_pred - Y_test), axis=0)),
          np.sum(np.mean((y_test_pred - Y_test) ** 2, axis=0)))

    print("sklearn train and test mae : ", median_absolute_error(y_pred, Y_train),
          median_absolute_error(Y_test, y_test_pred))


run_model('../TVP-VAR-VDA code/beta/BNB_amount_to_beta.csv', 10, 7)
# run_model('../TVP-VAR-VDA code/beta/BNB_amount_to_y_price_x_diff_beta.csv', 1, 14)
