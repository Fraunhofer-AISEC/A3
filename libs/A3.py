import tensorflow as tf
import random

from libs.regulariser import ShiftedL1L2

from typing import Union
from pathlib import Path


class A3:
    def __init__(
            self,
            target_network: tf.keras.Model = None,
            alarm_network: tf.keras.Model = None,
            anomaly_network: Union[tf.keras.Model, tf.keras.layers.Layer] = None,
            anomaly_loss_weight: float = 1.0,
            random_state: int = None
    ):
        # Config
        self.input_shape = None
        self.input_dim = None
        self.anomaly_loss_weight = anomaly_loss_weight
        self.random_state = random_state

        # Overall model
        self.target_network = None
        self.alarm_network = None
        self.anomaly_network = None
        self.overall_model: tf.keras.Model = None

        if target_network:
            self.add_target_network(target_network=target_network)
        if alarm_network:
            self.add_alarm_network(alarm_network=alarm_network)
        if anomaly_network:
            self.add_anomaly_network(anomaly_network=anomaly_network)

    def get_alarm_shape(self):
        """
        Get the shape needed for the alarm's input, i.e. the output shape of the target.
        :return: input shape for the alarm network
        """
        alarm_shape = self.extract_activations(self.target_network, self.target_network.name)

        alarm_shape = self.parse_shape(alarm_shape.shape)

        return alarm_shape

    def add_target_network(self, target_network: tf.keras.Model, freeze_model: bool = True):
        """
        Add q target network to the A3 system
        :param target_network: trained target model
        :param freeze_model: fix the model parameters to make them non-trainable
        :return:
        """
        if freeze_model:
            target_network.trainable = False

        self.target_network = target_network

        # Also save the input shape
        self.input_shape = self.parse_shape(self.target_network.input_shape)
        self.input_dim = self.shape_to_dim(self.input_shape)

    def add_alarm_network(self, alarm_network: tf.keras.Model):
        """
        Add an alarm network to the A3 system
        :param alarm_network: Keras model with the right input shape
        :return:
        """
        assert self.parse_shape(alarm_network.input_shape) == self.get_alarm_shape(), \
            "The input of the alarm network must have the shape of the target's output"

        self.alarm_network = alarm_network

        self._build_network()

    def add_anomaly_network(self, anomaly_network: Union[tf.keras.Model, tf.keras.layers.Layer], freeze_model: bool = True):
        """
        Add anomaly network to the A3 system
        :param anomaly_network: trained anomaly model
        :param freeze_model: fix the model parameters to make them non-trainable
        :return:
        """
        if freeze_model:
            anomaly_network.trainable = False

        self.anomaly_network = anomaly_network

    def _build_network(self):
        """
        Build graph from target and alarm network
        :return: concatenated target and alarm model
        """
        overall_input = self.target_network.inputs

        # Get activations for each target network
        all_activations = self.extract_activations(self.target_network, self.target_network.name)
        f_all_activations = tf.keras.models.Model(
            inputs=overall_input,
            outputs=all_activations,
            name="Activations"
        )

        # The alarm network takes care of the output
        out_alarm = self.alarm_network(all_activations)

        # Aggregate to overall target and alarm model
        self.overall_model = tf.keras.Model(
            inputs=overall_input,
            outputs=out_alarm,
            name="A3"
        )

        # If given, feed in the alarm network
        if self.anomaly_network:
            # Feed it through the A3 pipeline
            out_anomaly = self.anomaly_network.predict_anomaly(overall_input)
            a3_anomaly = f_all_activations(out_anomaly)
            a3_anomaly = self.alarm_network(a3_anomaly)

            # Calculate loss from this
            loss_anomaly = tf.keras.losses.binary_crossentropy(
                y_true=tf.keras.backend.ones_like(a3_anomaly),
                y_pred=a3_anomaly
            )
            # Reduce over all synthetic samples
            loss_anomaly = tf.keras.backend.mean(loss_anomaly, axis=0)
            loss_anomaly *= self.anomaly_loss_weight

            self.overall_model.add_loss(loss_anomaly)

    ## Keras functions
    def compile(self, optimizer="adam", loss="binary_crossentropy", *args, **kwargs):
        # Check if all models are given
        if self.target_network is None or self.alarm_network is None:
            raise AttributeError("Please specify an alarm and target network")

        # Build the overall model if not yet specified
        if self.overall_model is None:
            self._build_network()

        return self.overall_model.compile(optimizer=optimizer, loss=loss, *args, **kwargs)

    def fit(self, x, y, validation_data: tuple, epochs: int = 10, batch_size: int = 256,
            *args, **kwargs):
        return self.overall_model.fit(
            x=x, y=y,
            validation_data=validation_data, epochs=epochs, batch_size=batch_size, *args, **kwargs
        )

    def predict(self, x, get_activation: bool = False, *args, **kwargs):
        return self.overall_model.predict(x=x, *args, **kwargs)

    def extract_activations(
            self, target_network: tf.keras.Model, target_name: str,
            is_concatenated: bool = True
    ) -> tf.keras.layers.Layer:
        """
        Get the activation layers of the defined model
        :param target_network: model to take the activation layers from
        :param target_name: give the target a name
        :param is_concatenated: concatenate after extraction
        :return:
        """

        # Get all activations
        all_activations = [
            tf.keras.layers.Flatten(
                name=f"{target_name}_{i_layer}"
            )(cur_layer.output) for i_layer, cur_layer in enumerate(target_network.layers) if
            isinstance(cur_layer, tf.keras.layers.Activation)
        ]

        # Concatenate to one layer
        if is_concatenated:
            all_activations = tf.keras.layers.Concatenate(
                name=target_name
            )(all_activations)

        return all_activations

    @staticmethod
    def parse_shape(in_shape: tuple) -> tuple:
        """
        Parse the shape information, e.g. remove the batch size
        :param in_shape: raw shape, e.g. from a Keras layer
        :return: parsed shape usable as input
        """
        new_shape = in_shape[1:]

        if len(new_shape) == 1:
            new_shape = (new_shape[0],)

        return new_shape

    @staticmethod
    def shape_to_dim(in_shape: tuple) -> int:
        """
        Convert shape tuple to the overall dimensionality
        :param in_shape: parsed shape
        :return: overall dimension
        """
        total_dim = 1
        for cur_shape in in_shape:
            total_dim *= cur_shape

        return total_dim

    ## Load and save
    def save(self, basepath: Path, prefix: str = "", suffix: str = ".h5", *args, **kwargs):
        """
        Save target, alarm and overall network
        :param basepath: where to put all files
        :param prefix: prefix to all files
        :param args: arguments for Keras save function
        :param kwargs: arguments for Keras save function
        :return:
        """
        # Create the path if it does not exist
        if not basepath.exists():
            basepath.mkdir(parents=True, exist_ok=False)

        # Save target
        self.target_network.save(filepath=(basepath / prefix).with_suffix(f".target{suffix}"), *args, **kwargs)
        # Save anomaly
        if self.anomaly_network:
            try:
                self.anomaly_network.save_weights(filepath=str((basepath / prefix).with_suffix(f".anomaly{suffix}")), *args, **kwargs)
            except AttributeError:
                print("Cannot save the anomaly network. May be ignored if it is only a layer.")
        # Save alarm
        self.alarm_network.save(filepath=(basepath / prefix).with_suffix(f".alarm{suffix}"), *args, **kwargs)

        # There is a bug in TF<2.2 which causes the save operation to fail for .h5 files
        # Dirty hack: append random number to all layers
        # https://github.com/tensorflow/tensorflow/issues/32672
        for cur_layer in self.overall_model.layers:
            cur_layer._name = f'{cur_layer.name}_rand{random.randint(10, 100)}'

        return self.overall_model.save_weights(
            filepath=str((basepath / prefix).with_suffix(f".overall{suffix}")),
            *args, **kwargs
        )

    def load_target(self, data_path: Path, *args, **kwargs):
        """
        Load target
        :param data_path: path to target
        :param args: arguments for Keras load
        :param kwargs: arguments for Keras load
        :return:
        """

        try:
            target_network = tf.keras.models.load_model(
                filepath=data_path,
                *args, **kwargs)
        except (FileNotFoundError, OSError):
            raise FileNotFoundError(f"Cannot find target network at {data_path}.")
        # Had conflict of names -> rename target model
        target_network._name = 'target'

        self.add_target_network(target_network)

    def load_anomaly(
            self, data_path: Path, anomaly_model: Union[tf.keras.layers.Layer, tf.keras.Model], *args, **kwargs
    ):
        """
        Load target
        :param data_path: path to target
        :param anomaly_model: the bare anomaly model
        :param args: arguments for Keras load
        :param kwargs: arguments for Keras load
        :return:
        """

        try:
            anomaly_model.load_weights(
                filepath=data_path,
                *args, **kwargs)
            anomaly_model._name = 'anomaly'
        except AttributeError:
            # It's also fine if we just pass a layer
            pass

        self.add_anomaly_network(anomaly_model)

    def load_all(
            self, basepath: Path,
            prefix: str = "", suffix: str = ".h5", *args, **kwargs
    ):
        """
        Load target, alarm and overall network
        :param basepath: where all files are located
        :param prefix: prefix to all files
        :param args: arguments for Keras save function
        :param kwargs: arguments for Keras save function
        :return:
        """
        # Load target
        self.load_target(data_path=(basepath / prefix).with_suffix(f".target{suffix}"))

        # Load anomaly
        try:
            self.load_anomaly(
                data_path=(basepath / prefix).with_suffix(f".anomaly{suffix}"),
                anomaly_model=self.anomaly_network,
                *args, **kwargs
            )
        except FileNotFoundError:
            print("No anomaly network found. Ignored.")
            pass

        # Load alarm
        try:
            alarm_network = tf.keras.models.load_model(
                filepath=(basepath / prefix).with_suffix(f".alarm{suffix}"), custom_objects={
                    "ShiftedL1L2": ShiftedL1L2,
                }, *args, **kwargs
            )
            self.add_alarm_network(alarm_network)
        except FileNotFoundError:
            pass

        # We may build the model if the alarm model is given
        if self.alarm_network is not None:
            self._build_network()

        # Load overall model weights
        try:
            self.overall_model.load_weights(
                filepath=str((basepath / prefix).with_suffix(f".overall{suffix}")),
                *args, **kwargs
            )
        except AttributeError:
            self.overall_model = None
