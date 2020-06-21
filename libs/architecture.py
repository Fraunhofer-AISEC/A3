import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from libs.NeuralNet import NeuralNet
from libs.regulariser import ShiftedL1L2

# -- Target Networks --
def dense_ae(
        input_shape: tuple, layer_dims: List[int], hidden_activation="relu", p_dropout=.1, name="dense_ae", *args, **kwargs
) -> tf.keras.Model:
    """
    Simple, symmetric autoencoder structure
    :param input_shape: input dimension
    :param layer_dims: dimension of the autoencoder from input to code (the rest is automatically added)
    :param hidden_activation: activation of the hidden layers (output will be a sigmoid)
    :param p_dropout: dropout rate
    :param args: passed to autoencoder builder
    :param kwargs: passed to autoencoder builder
    :return: autoencoder model
    """
    # Construct simple autoencoder network
    network = tf.keras.Sequential(name=name)

    # Add symmetric encoder/decoder structure
    NeuralNet.add_symmetric_autoencoder(
        network, layer_dims=layer_dims, input_shape=input_shape, activation=hidden_activation, *args, **kwargs
    )
    # Add a dropout layer TODO: what about a regulariser?
    network.add(tf.keras.layers.Dropout(p_dropout))
    # Get an output of the same dimension as the input
    network.add(tf.keras.layers.Dense(input_shape[-1], activation="sigmoid"))

    return network


def conv_ae(input_shape: tuple, hidden_activation="relu", hidden_padding="same", p_dropout=.1) -> tf.keras.Model:
    """
    Convolutional autoencoder as proposed by
    https://blog.keras.io/building-autoencoders-in-keras.html
    :param input_shape: input dimension
    :param layer_dims: dimension of the autoencoder from input to code (the rest is automatically added)
    :param hidden_activation: activation of the hidden layers (output will be a sigmoid)
    :param p_dropout: dropout rate
    :return: convolutional autoencoder model
    """
    input_img = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding=hidden_padding)(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding=hidden_padding)(x)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding=hidden_padding)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding=hidden_padding)(x)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding=hidden_padding)(x)
    # x = tf.keras.layers.SpatialDropout2D(p_dropout)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding=hidden_padding)(x)
    encoded = tf.keras.layers.Activation(hidden_activation)(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = tf.keras.layers.Conv2D(8, (3, 3), padding=hidden_padding)(encoded)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding=hidden_padding)(x)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3))(x)
    x = tf.keras.layers.SpatialDropout2D(p_dropout)(x)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding=hidden_padding, name='target_output')(x)

    return tf.keras.Model(input_img, decoded)


def conv_vae(input_shape: tuple, hidden_activation="relu", hidden_padding="same", p_dropout=.1) -> tf.keras.Model:
    """
    Convolutional autoencoder as proposed by
    https://blog.keras.io/building-autoencoders-in-keras.html
    :param input_shape: input dimension
    :param layer_dims: dimension of the autoencoder from input to code (the rest is automatically added)
    :param hidden_activation: activation of the hidden layers (output will be a sigmoid)
    :param p_dropout: dropout rate
    :return: convolutional autoencoder model
    """
    input_img = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding=hidden_padding)(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding=hidden_padding)(x)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding=hidden_padding)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding=hidden_padding)(x)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding=hidden_padding)(x)
    # x = tf.keras.layers.SpatialDropout2D(p_dropout)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding=hidden_padding)(x)
    encoded = tf.keras.layers.Activation(hidden_activation)(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = tf.keras.layers.Conv2D(8, (3, 3), padding=hidden_padding)(encoded)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding=hidden_padding)(x)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3))(x)
    x = tf.keras.layers.SpatialDropout2D(p_dropout)(x)
    x = tf.keras.layers.Activation(hidden_activation)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding=hidden_padding, name='target_output')(x)

    return tf.keras.Model(input_img, decoded)


# -- Anomaly Networks --
class RandomNoise(tf.keras.layers.Layer):
    def __init__(self, noise_type: str = "normal"):
        """
        Layer that replaces the input by random noise
        :param noise_type: either "normal" or "uniform"
        """
        super(RandomNoise, self).__init__()
        self.noise_type = noise_type

    def call(self, inputs):
        first_input = inputs[0]

        # We just need noise. To get some proper gradient, we'll multiply the input by zero.
        zero_input = tf.keras.backend.zeros_like(first_input, name="noise_zero_helper")

        if self.noise_type == "uniform":
            noise_add = tf.keras.backend.random_uniform(
                shape=tf.keras.backend.shape(first_input),
                minval=0, maxval=1,
            )
        elif self.noise_type == "normal":
            noise_add = tf.keras.backend.random_normal(
                shape=tf.keras.backend.shape(first_input),
                mean=.5, stddev=1.0
            )
        else:
            TypeError("Noise type must be 'normal' or 'uniform'")

        noise_out = tf.keras.layers.Multiply()([first_input, zero_input])
        noise_out = tf.keras.layers.Add()([noise_out, noise_add])

        return noise_out

    def predict_anomaly(self, inputs):
        return self.call(inputs)

# Variational AutoEncoder based on
# https://www.tensorflow.org/guide/keras/custom_layers_and_models (CC BY 4.0 license)
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, std_dev=1.0):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), stddev=std_dev)

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(
            self, layer_dims, name="encoder", **kwargs
    ):
        super(Encoder, self).__init__(name=name, **kwargs)

        # Save config
        self.layer_dims = layer_dims
        code_dim = layer_dims[-1]

        # Build model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
        ])
        NeuralNet.add_dense(
            self.model, layer_dims=self.layer_dims[:-1], activation="relu"
        )

        self.dense_mean = tf.keras.layers.Dense(code_dim)
        self.dense_log_var = tf.keras.layers.Dense(code_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.model(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def get_config(self):
        config = {
            "layer_dims": self.layer_dims,
            "name": self.name
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(tf.keras.layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(
            self, original_dim, layer_dims, name="decoder", p_dropout=.1, **kwargs
    ):
        super(Decoder, self).__init__(name=name, **kwargs)

        # Save config
        self.original_dim = original_dim
        self.original_layer_dims = layer_dims
        self.layer_dims = list(reversed(layer_dims[:-1]))
        code_dim = layer_dims[-1]
        self.p_dropout = p_dropout

        # Build model
        original_dim_product = 1
        for cur_el in original_dim:
            original_dim_product *= cur_el
        # self.dense_proj = tf.keras.layers.Dense(code_dim, activation="relu")
        self.dense_output = tf.keras.models.Sequential()
        NeuralNet.add_dense(
            self.dense_output, layer_dims=self.layer_dims, activation="relu"
        )
        self.dense_output.add(tf.keras.layers.Dropout(self.p_dropout))
        self.dense_output.add(tf.keras.layers.Dense(original_dim_product, activation="sigmoid"))
        self.dense_output.add(tf.keras.layers.Reshape(original_dim))

    def call(self, inputs):
        # x = self.dense_proj(inputs)
        return self.dense_output(inputs)

    def get_config(self):
        config = {
            "original_dim": self.original_dim,
            "layer_dims": self.original_layer_dims,
            "name": self.name,
            "p_dropout": self.p_dropout
        }
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
            self, input_shape, layer_dims, anomaly_var=5.0, name='autoencoder', **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)

        # VAE components
        self.encoder = Encoder(layer_dims=layer_dims)
        self.decoder = Decoder(input_shape, layer_dims=layer_dims)
        self.sampling = Sampling()
        self.anomaly_var = anomaly_var

        # Save dimensionality
        self.original_layer_dims = layer_dims
        self.original_dim = input_shape
        self.original_dim_product = 1
        for cur_el in input_shape:
            self.original_dim_product *= cur_el

    def predict_anomaly(self, inputs):
        # Encode the input
        z_mean, z_log_var, z = self.encoder(inputs[0])
        # Sample around the mean
        zero_var = tf.keras.backend.zeros_like(z_mean)
        z_log_var = tf.keras.layers.Multiply()([zero_var, z_log_var])
        z = self.sampling([z_mean, z_log_var], std_dev=self.anomaly_var)
        # Reconstruct picture
        reconstructed = self.decoder(z)

        return reconstructed

    def call(self, inputs):
        # Intermediate results
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Objective 1: match the labels
        match_loss = tf.keras.losses.binary_crossentropy(
            inputs, reconstructed
        )
        match_loss = tf.keras.backend.batch_flatten(match_loss)
        match_loss = tf.keras.backend.sum(match_loss, axis=-1)

        # Objective 2: distribution should be normal
        kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # Fulfill both objectives, TODO: add weighting?
        overall_loss = tf.keras.backend.mean(match_loss + kl_loss)
        self.add_loss(overall_loss)

        return reconstructed

    def plot_latent(self, inputs, x_axis=0, y_axis=1, color_labels=None):
        # Show how the means distribute
        z_mean, _, _ = self.encoder(inputs)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, x_axis], z_mean[:, y_axis], c=color_labels)
        plt.colorbar()
        plt.xlabel(f"z[{x_axis}]")
        plt.ylabel(f"z[{y_axis}]")
        plt.show()

    def plot_mnist(self, inputs):
        # This code piece was copied from
        # https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py (MIT license)
        z_mean, _, _ = self.encoder(inputs)
        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-5, 5, n)
        grid_y = np.linspace(-5, 5, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder(z_sample).numpy()
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = (n - 1) * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

    def get_config(self):
        config = {
            "input_shape": self.original_dim,
            "layer_dims": self.original_layer_dims,
            "anomaly_var": self.anomaly_var,
            "name": self.name,
        }
        base_config = super(VariationalAutoEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# -- Alarm Networks --
def alarm_net(
        input_shape: tuple, layer_dims: list, hidden_activation="relu", p_dropout=.1,
        in_l1: float = 0.0, in_l2: float = 0.0, out_l1: float = 0.0, out_l2: float = 0.0, *args, **kwargs
):
    # Construct simple deep network
    network = tf.keras.Sequential()
    NeuralNet.add_dense(
        network, layer_dims=layer_dims, input_shape=input_shape, activation=hidden_activation,
        first_l1=in_l1, first_l2=in_l2, *args, **kwargs
    )
    # Add a dropout layer
    network.add(tf.keras.layers.Dropout(p_dropout))
    # Get a binary output (anomaly or not)
    network.add(tf.keras.layers.Dense(
        1, activation="sigmoid", activity_regularizer=ShiftedL1L2(new_zero=1, l1=out_l1, l2=out_l2))
    )

    return network
