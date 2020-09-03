from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow.keras import backend as K


def get_model(input_shape, output_dim):
    model = Sequential()
    model.add(tf.keras.Input(shape=(input_shape[1],)))
    model.add(Dense(50))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Dense(6))
    return model


# This class has been inspired from the paper https://arxiv.org/pdf/1703.07015.pdf
class PreAutoRegressionTransformation(tf.keras.layers.Layer):
    def __init__(self, lag_ar, **kwargs):
        # Number of timeseries values to consider for the linear layer (AR layer)
        self.lag_ar = lag_ar
        super(PreAutoRegressionTransformation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PreAutoRegressionTransformation, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        batchsize = tf.shape(x)[0]
        # shape of the input data
        input_shape = K.int_shape(x)
        # Select only lagged inputs
        output = x[:, -self.lag_ar:, :]
        output = tf.transpose(output, perm=[0, 2, 1])
        output = tf.reshape(output, [batchsize * input_shape[2], self.lag_ar])
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'lag_ar': self.lag_ar,
        })
        return config


# This class has been inspired from the paper https://arxiv.org/pdf/1703.07015.pdf
class PostAutoRegressionTransformation(tf.keras.layers.Layer):
    def __init__(self, n_dim, **kwargs):
        # Number of timeseries
        self.n_dim = n_dim
        super(PostAutoRegressionTransformation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PostAutoRegressionTransformation, self).build(input_shape)

    def call(self, inputs):
        x, original_model_input = inputs
        batchsize = tf.shape(original_model_input)[0]
        output = tf.reshape(x, [batchsize, self.n_dim])
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_dim': self.n_dim,
        })
        return config


def get_TVP_VARNet_model(input_shape, output_dim):
    X = tf.keras.Input(shape=input_shape[1:])
    out4, _, _ = tf.keras.layers.LSTM(100, return_state=True, return_sequences=False, activation='relu',
                                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0))(X)
    out41 = tf.keras.layers.Flatten()(out4)
    out5 = tf.keras.layers.Dense(output_dim)(out41)

    new_input = X
    Z = PreAutoRegressionTransformation(10)(new_input)
    Z = Flatten()(Z)
    Z = Dense(1)(Z)
    Z = PostAutoRegressionTransformation(output_dim)([Z, new_input])
    print("Z shape: ", Z.shape)
    Y = Add()([out5, Z])
    model = tf.keras.Model(inputs=X, outputs=Y)
    return model