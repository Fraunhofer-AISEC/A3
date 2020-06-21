import abc
import tensorflow as tf
import numpy as np

from typing import List, Union, Tuple
from pathlib import Path

# Note: this library was used in a previous version which we omitted in favour of plain Keras Models.
# We still use some of the static methods, though.
class NeuralNet:
    # General class defining useful neural network operations
    def __init__(self, input_shape: tuple, predefined_nn: tf.keras.models.Model = None, *args, **kwargs):
        """
        Prepare neural network
        :param input_shape: input shape
        :param args: passed to the network generator
        :param kwargs: passed to the network generator
        """

        # Note down the input shape: first dimension is variable based on the sample length
        self.input_shape = input_shape  # if input_shape[0] is None else (None, )+input_shape

        # Either generate a model or just take what was given
        self.nn = predefined_nn if predefined_nn is not None else self._get_network(*args, **kwargs)

    @abc.abstractmethod
    def _get_network(self, *args, **kwargs) -> tf.keras.Sequential:
        """
        Network structure to be implemented by subclasses
        :return: Keras sequential or functional model
        """
        pass

    # == Function Mappers ==
    def compile(
            self,
            optimiser: Union[str, tf.keras.optimizers.Optimizer],
            loss: Union[str, tf.keras.losses.Loss],
            metrics: Union[str, List[str], tf.keras.metrics.Metric],
            *args, **kwargs
    ):
        self.nn.compile(optimizer=optimiser, loss=loss, metrics=metrics)

    def fit(
            self, x, y, epochs=15, batch_size=32, validation_split=0.0, validation_data=None, verbose=2, *args, **kwargs
    ):
        return self.nn.fit(
            x=x, y=y,
            epochs=epochs, batch_size=batch_size, validation_split=validation_split, validation_data=validation_data,
            verbose=verbose, *args, **kwargs
        )

    def predict(self, *args, **kwargs):
        return self.nn.predict(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.nn.train(*args, **kwargs)

    def train_on_batch(self, *args, **kwargs):
        return self.nn.train_on_batch(*args, **kwargs)

    def save(self, filepath: Path, *args, **kwargs):
        # Create the path if it does not exist
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True, exist_ok=False)

        return self.nn.save(filepath=filepath, *args, **kwargs)

    # == Network Generators ==
    @staticmethod
    def add_dense(
            network: tf.keras.Sequential, layer_dims: List[int], input_shape=None, activation="relu",
            first_l1: float = 0.0, first_l2: float = 0.0, p_dropout: float = None, *args, **kwargs
    ):
        """
        Build a dense model with the given hidden state
        :param network: sequential Keras network
        :param layer_dims: list of hidden state dimensions
        :param first_l1: L1 kernel regulariser on the first layer
        :param first_l2: L2 kernel regulariser on the first layer
        :param p_dropout: dropout percentage after the first layer
        :param args: passed to Keras dense layer
        :param kwargs: passed to Keras dense layer
        :return: sequential Keras dense model
        :return:
        """

        # First layer
        if input_shape:
            network.add(tf.keras.layers.Dense(
                layer_dims[0], input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=first_l1, l2=first_l2),
                bias_regularizer=tf.keras.regularizers.L1L2(l1=first_l1, l2=first_l2),
            ))
        else:
            network.add(tf.keras.layers.Dense(
                layer_dims[0],
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=first_l1, l2=first_l2),
                bias_regularizer=tf.keras.regularizers.L1L2(l1=first_l1, l2=first_l2),
            ))
        network.add(tf.keras.layers.Activation(activation))
        if p_dropout:
            network.add(tf.keras.layers.Dropout(p_dropout))
        # All the other feature_layers
        for cur_dim in layer_dims[1:]:
            network.add(tf.keras.layers.Dense(cur_dim, *args, **kwargs))
            network.add(tf.keras.layers.Activation(activation))

    @staticmethod
    def add_symmetric_autoencoder(
            network: tf.keras.Sequential, layer_dims: List[int], input_shape=None, activation="relu", *args, **kwargs
    ) -> tf.keras.Sequential:
        """
        Build autoencoder where the hidden state dimensions of the en- and decoder are the same
        :param network: sequential Keras network
        :param layer_dims: list of hidden state dimensions
        :param args: passed to Keras dense layer
        :param kwargs: passed to Keras dense layer
        :return: sequential Keras autoencoder model
        """

        # First layer
        network.add(tf.keras.layers.Dense(layer_dims[0], input_shape=input_shape))
        network.add(tf.keras.layers.Activation(activation))
        # Encoder
        for cur_dim in layer_dims[1:]:
            network.add(tf.keras.layers.Dense(cur_dim, *args, **kwargs))
            network.add(tf.keras.layers.Activation(activation))
        # Decoder
        for cur_dim in reversed(layer_dims[:-1]):
            network.add(tf.keras.layers.Dense(cur_dim, *args, **kwargs))
            network.add(tf.keras.layers.Activation(activation))
