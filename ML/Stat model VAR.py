import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt


def preprocess_data():
    with open('../TVP-VAR-VDA code/beta/BNB_amount_to_beta.npy', 'rb') as f:
        b_var = np.load(f)
    b_var = np.reshape(b_var, (b_var.shape[0], b_var.shape[1]))[:, 6:12]

    return b_var * 100


def create_dataset(Betas, time_steps=2):
    Xs, ys = [], []
    for i in range(len(Betas) - time_steps):
        v = Betas[i:(i + time_steps)]
        Xs.append(v)
        ys.append(Betas[i + time_steps])
    return np.array(Xs), np.array(ys)


def run_model(input):
    model = VAR(input)
    result = model.fit(20)
    out2 = []
    for i in range(out.shape[0]):
        out2.append(result.forecast(in_p[i], 1))
    out21 = np.array(out2)
    arr_out = np.reshape(out21, (out21.shape[0], out21.shape[2])) * 10
    return arr_out


def plot_pred(pred, out):
    plt.plot(pred[:, 1], linestyle="--", label="predicted")
    plt.plot(out[:, 1], label="true")
    plt.legend()
    plt.savefig('VAR_train')


X = preprocess_data()
in_p, out = create_dataset(X, 20)
predictions = run_model(X)
plot_pred(predictions, out)
