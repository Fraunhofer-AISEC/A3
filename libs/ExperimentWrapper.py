import random
import re
import math

from dataclasses import dataclass
import numpy as np
import tensorflow as tf

from typing import Callable, Union, List, Any
from pathlib import Path
from warnings import warn

from libs.A3 import A3
from libs.DataHandler import DataLabels, MNIST
from libs.NeuralNet import NeuralNet
from utils import BASE_PATH

from joblib import dump
from sklearn.metrics import precision_score, recall_score, f1_score

@dataclass
class ExperimentData:
    """
    All data needed for the experiment
    """
    train_target: tuple
    train_classifier: tuple
    train_alarm: tuple
    val_target: tuple
    val_classifier: tuple
    val_alarm: tuple
    test_target: tuple
    test_classifier: tuple
    test_alarm: tuple
    data_shape: tuple

@dataclass
class ExperimentConfig:
    """
    Configuration which data is used in the respective experiment
    """
    data_set: DataLabels  # Data set to use
    train_normal: list  # Classes for normal samples
    train_anomaly: list  # Classes for known anomalies
    test_anomaly: list  # Classes for test anomalies

    def to_data(
            self, train_type: str = "train", test_type: str = "test", n_anomaly_samples: int = None
    ) -> ExperimentData:
        """
        Convert the configuration to actual data
        :param train_type: use the train or validation data for training (only used to load less data while debugging)
        :param test_type: use the test or validation data for evaluation (i.e. code once, use twice)
        :param n_anomaly_samples: limit the number of anomaly samples in the training data
        """
        return ExperimentData(
            # Target training: all normal samples
            train_target=self.data_set.get_target_autoencoder_data(
                data_split=train_type, include_classes=self.train_normal
            ),
            train_classifier=self.data_set.get_target_classifier_data(
                data_split=train_type, include_classes=self.train_normal
            ),
            # Alarm training: all normal samples plus the ones known to be anomalous
            train_alarm=self.data_set.get_alarm_data(
                data_split=train_type, include_classes=list(set(self.train_normal) | set(self.train_anomaly)),
                anomaly_classes=self.train_anomaly, n_anomaly_samples=n_anomaly_samples
            ),
            # Target validation: all normal samples plus the ones that should also be anomalous while training
            val_target=self.data_set.get_target_autoencoder_data(
                data_split="val", include_classes=list(set(self.train_normal) | set(self.train_anomaly))
            ),
            val_classifier=self.data_set.get_target_classifier_data(
                data_split="val", include_classes=list(set(self.train_normal) | set(self.train_anomaly))
            ),
            val_alarm=self.data_set.get_alarm_data(
                data_split="val", include_classes=list(set(self.train_normal) | set(self.train_anomaly)),
                anomaly_classes=self.train_anomaly
            ),
            # Target testing: all normal samples plus the test anomalies
            test_target=self.data_set.get_target_autoencoder_data(
                data_split=test_type, include_classes=list(set(self.train_normal) | set(self.test_anomaly))
            ),
            test_classifier=self.data_set.get_target_classifier_data(
                data_split=test_type, include_classes=list(set(self.train_normal) | set(self.test_anomaly))
            ),
            test_alarm=self.data_set.get_alarm_data(
                data_split=test_type, include_classes=list(set(self.train_normal) | set(self.test_anomaly)),
                anomaly_classes=self.test_anomaly
            ),
            # Shape to generate networks
            data_shape=self.data_set.shape
        )


class ExperimentWrapper:
    def __init__(
            self, data_setup: List[ExperimentConfig], n_anomaly_samples: List[int],
            save_prefix: str = '', random_seed: int = None, is_override=False,
            target_folder="", anomaly_folder="", alarm_folder="", train_type: str = "train", test_type: str = "test",
            out_path: Path = BASE_PATH
    ):
        """
        Wrapper class to have a common scheme for the experiments
        :param data_setup: data configuration for every experiment
        :param n_anomaly_samples: anomaly samples while training
        :param save_prefix: prefix for saved NN models
        :param random_seed: seed to fix the randomness
        :param is_override: override output if already exists
        :param target_folder: subfolder for target networks if pre-trained models are used
        :param anomaly_folder: subfolder for anomaly networks if pre-trained models are used
        :param alarm_folder: subfolder for alarm networks if pre-trained models are used
        :param train_type: use the train or validation data for training (only used to load less data while debugging)
        :param test_type: use the test or validation data for evaluation
        :param out_path: output base path for the models, usually the base path
        """

        # Get a parameter grid
        self.data_setup = data_setup
        self.n_anomaly_samples = n_anomaly_samples

        # Configuration
        self.is_override = is_override
        self.train_type = train_type
        self.test_type = test_type

        # Folder paths
        self.out_path = out_path
        self.target_folder = target_folder
        self.anomaly_folder = anomaly_folder
        self.alarm_folder = alarm_folder
        self.save_prefix = save_prefix
        if random_seed is not None:
            self.save_prefix += f"_{random_seed}"

        # Fix randomness
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        # Alright, we can't make the NN deterministic on a GPU [1]. Probably makes more sense to keep the sample
        # selection deterministic, but repeat all NN-related aspects.
        # [1] https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        # tf.random.set_seed(random_seed)

    def train_target(
            self, target_net: Callable[[int], tf.keras.Model],
            network_params: dict = {}, compile_params: dict = {}, fit_params: dict = {},
            is_subfolder=True,
    ):
        """
        Train the target network, e.g. the original autoencoder
        :param target_net: callable neural network used for the target
        :param network_params: additional arguments passed to the network class
        :param compile_params: additional arguments passed to the network class
        :param fit_params: additional arguments passed to the network class
        :param is_subfolder: create a subfolder for each network architecture
        :return:
        """

        for cur_data in self.data_setup:
            # print(f"Now training target {self.parse_name(cur_data)} for {cur_n_anomaly} anomaly samples")
            print(f"Now training target {self.parse_name(cur_data)}")

            # Check if model exists - note that we need to retrain the target each time as the data split changes
            out_path = self.get_model_path(
                    base_path=self.out_path, file_name=self.parse_name(cur_data),
                    sub_folder="target", sub_sub_folder=self.parse_name(network_params) if is_subfolder else ""
                )
            if not self.is_override and out_path.exists():
                print("This target was already trained. Use is_override=True to override it.")
                continue

            # Load data
            this_data = cur_data.to_data(
                train_type=self.train_type, test_type=self.test_type
            )

            # Create the target network
            this_net = target_net(input_shape=cur_data.data_set.shape, **network_params)

            # We need a fresh optimizer for each target
            compile_optimizer = "adam"
            if "optimizer" in compile_params:
                raise NotImplementedError("Please omit the 'optimizer' keyword. So far only Adam is supported.")
            if "learning_rate" in compile_params:
                compile_optimizer = tf.keras.optimizers.Adam(compile_params["learning_rate"])
                del compile_params["learning_rate"]
            this_net.compile(optimizer=compile_optimizer, **compile_params)

            this_net.fit(
                x=this_data.train_target[0], y=this_data.train_target[1],
                validation_data=this_data.val_target,
                **fit_params
            )

            # Save the target model
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=False)
            this_net.save(out_path)
            self.target_folder = self.parse_name(network_params) if is_subfolder else ""

    def train_anomaly(
            self, anomaly_net: Callable[[int], tf.keras.Model],
            network_params: dict = {}, compile_params: dict = {}, fit_params: dict = {},
            is_subfolder=True,
    ):
        """
        Train the anomaly network, e.g. a VAE
        :param anomaly_net: callable neural network used for the anomaly
        :param network_params: additional arguments passed to the network class
        :param compile_params: additional arguments passed to the network class
        :param fit_params: additional arguments passed to the network class
        :param is_subfolder: create a subfolder for each network architecture
        :return: last trained net as example for the overall model to load the weights into
        """

        # As we save weights (because SavedModel somehow does not restore the member functions), we need some base model
        this_net = anomaly_net(input_shape=self.data_setup[0].data_set.shape, **network_params)

        for cur_data in self.data_setup:
            print(f"Now training anomaly {self.parse_name(cur_data)}")

            # Check if model exists - note that we need to retrain the target each time as the data split changes
            out_path = self.get_model_path(
                    base_path=self.out_path, file_name=self.parse_name(cur_data),
                    sub_folder="anomaly", sub_sub_folder=self.parse_name(network_params) if is_subfolder else ""
                )
            if not self.is_override and out_path.exists():
                print("This anomaly network was already trained. Use is_override=True to override it.")
                continue

            # Load data
            this_data = cur_data.to_data(
                train_type=self.train_type, test_type=self.test_type
            )

            # Create the target network
            this_net = anomaly_net(input_shape=cur_data.data_set.shape, **network_params)

            # We need a fresh optimizer for each target
            compile_optimizer = "adam"
            if "optimizer" in compile_params:
                raise NotImplementedError("Please omit the 'optimizer' keyword. So far only Adam is supported.")
            if "learning_rate" in compile_params:
                compile_optimizer = tf.keras.optimizers.Adam(compile_params["learning_rate"])
                del compile_params["learning_rate"]
            this_net.compile(optimizer=compile_optimizer, **compile_params)

            this_net.fit(
                x=this_data.train_target[0], y=this_data.train_target[1],
                validation_data=this_data.val_target,
                **fit_params
            )

            # Save the target model
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=False)
            this_net.save_weights(str(out_path))
            self.anomaly_folder = self.parse_name(network_params) if is_subfolder else ""

        return this_net

    def train_a3(
            self, alarm_net: Callable[[Any], tf.keras.Model], epoch_dict: dict = None,
            anomaly_net: Union[tf.keras.Model, tf.keras.layers.Layer] = None,
            target_path: Path = None,
            network_params: dict = {}, compile_params: dict = {}, fit_params: dict = {},
            is_subfolder: bool = False, show_metrics: bool = False, automatic_weights: bool = True
    ):
        """
        Train the alarm network, e.g. the anomaly classifier
        :param alarm_net: callable neural network used for the alarm net
        :param epoch_dict: defines the number of epochs per n_anomalies
        :param anomaly_net: anomaly network to generate anomalies
        :param target_path: path to a target network that it used instead of the ones trained with train_target()
        :param network_params: additional arguments passed to the network class
        :param compile_params: additional arguments passed to the network class
        :param fit_params: additional arguments passed to the network class
        :param is_subfolder: create a subfolder for each network architecture
        :param show_metrics: show the metrics (F1 etc) after each epoch
        :param automatic_weights: calculate the class weights based on the class imbalance, if False take 1
        :return:
        """

        for cur_data in self.data_setup:
            for cur_n_anomaly in self.n_anomaly_samples:
                print(f"Now training alarm {self.parse_name(cur_data)} for {cur_n_anomaly} anomaly samples")

                # Check if model exists
                out_path = self.get_model_path(
                    base_path=self.out_path, file_name=self.parse_name(cur_data, additional_info=f"anom={cur_n_anomaly}"), sub_folder="a3",
                    sub_sub_folder=self.parse_name(network_params) if is_subfolder else ""
                )
                if not self.is_override and out_path.with_suffix(".overall.h5").exists():
                    print("This A3 model was already trained. Use is_override=True to override it.")
                    continue

                # Load anomaly & target to overall model
                model_a3 = A3()
                model_a3.load_anomaly(
                    self.get_model_path(
                        base_path=self.out_path, file_name=self.parse_name(cur_data), sub_folder="anomaly",
                        sub_sub_folder=self.target_folder
                    ),
                    anomaly_model=anomaly_net
                )
                model_a3.load_target(
                    self.get_model_path(
                        base_path=self.out_path, file_name=self.parse_name(cur_data), sub_folder="target",
                        sub_sub_folder=self.target_folder
                    ) if not target_path else target_path
                )

                # Load data
                this_data = cur_data.to_data(
                        train_type=self.train_type, test_type=self.test_type, n_anomaly_samples=cur_n_anomaly
                    )

                # Construct the alarm network and train
                this_net = alarm_net(input_shape=model_a3.get_alarm_shape(), **network_params)
                model_a3.add_alarm_network(this_net)

                # TODO: we could also reinitiate the optimizer calling __init__() again, couldn't we?
                compile_optimizer = "adam"
                if "optimizer" in compile_params:
                    raise NotImplementedError("Please omit the 'optimizer' keyword. So far only Adam is supported.")
                if "learning_rate" in compile_params:
                    compile_optimizer = tf.keras.optimizers.Adam(compile_params["learning_rate"])

                # Compile the alarm model
                model_a3.compile(
                    optimizer=compile_optimizer,
                    **{
                        cur_key: cur_val for cur_key, cur_val in compile_params.items() if cur_key != "learning_rate"
                    }
                )

                # Add callbacks if we like to see some metrics
                all_cb = []
                if show_metrics:
                    these_metrics = Metrics(training_data=this_data.train_alarm, validation_data=this_data.val_alarm)
                    all_cb.append(these_metrics)

                # Calculate appropriate class weights
                anom_weight = 1 if not automatic_weights else\
                    self.automatic_class_weights(train_data=this_data.train_alarm, n_anomalies=cur_n_anomaly)
                # Alter the number of epochs if given
                if epoch_dict:
                    fit_params["epochs"] = epoch_dict[cur_n_anomaly]

                model_a3.fit(
                    x=this_data.train_alarm[0], y=this_data.train_alarm[1],
                    validation_data=this_data.val_alarm, callbacks=all_cb, class_weight={0: 1, 1: anom_weight},
                    **fit_params
                )

                # Save the model
                model_a3.save(out_path.parent, prefix=out_path.name)
                self.alarm_folder = self.parse_name(network_params) if is_subfolder else ""

    # -- Baselines --
    def train_baseline(
            self, is_subfolder=True, baseline: str = '', **kwargs
    ):
        """
        Train some baseline methods, e.g. isolation forest
        :param is_subfolder: create a subfolder for each network architecture
        :param baseline: which baseline method to evaluate
        :param kwargs: extra arguments for the baseline method
        :return:
        """

        for cur_data in self.data_setup:
            print(f"Now training baseline method '{baseline}' for {self.parse_name(cur_data)}")

            # Check if the respective model exists
            if baseline == 'DAGMM':
                file_suffix = ""
            elif baseline == "DevNet":
                file_suffix = ".h5"
            else:
                file_suffix = ".joblib"

            out_path = self.get_model_path(
                base_path=self.out_path, file_name=self.parse_name(cur_data), file_suffix=file_suffix,
                sub_folder=baseline, sub_sub_folder=""
            )
            if not self.is_override and out_path.exists():
                print("This baseline method was already trained. Use is_override=True to override it.")
                continue

            # The baselines are semi-supervised in the sense they only need normal data. We give the fully
            # semi-supervised methods the advantage of having the maximum amount of anomalous examples.
            else:
                this_data = cur_data.to_data(
                    train_type=self.train_type, test_type=self.test_type, n_anomaly_samples=max(self.n_anomaly_samples)
                )

            # Fit the baseline method
            try:
                if baseline == 'IsolationForest':
                    from sklearn.ensemble import IsolationForest
                    baseline_model = IsolationForest(random_state=self.random_seed, n_jobs=3, **kwargs)
                    baseline_model.fit(this_data.train_target[0].reshape(this_data.train_target[0].shape[0], -1))
                elif baseline == 'OC-SVM':
                    from sklearn.svm import OneClassSVM
                    baseline_model = OneClassSVM(max_iter=100, verbose=True, **kwargs)
                    baseline_model.fit(
                        this_data.train_target[0].reshape(this_data.train_target[0].shape[0], -1)
                    )
                elif baseline == 'DAGMM':
                    from baselines.dagmm import DAGMM
                    baseline_model = DAGMM(random_seed=self.random_seed, **kwargs)
                    baseline_model.fit(this_data.train_target[0].reshape(this_data.train_target[0].shape[0], -1))
                elif baseline == "DevNet":
                    from baselines.devnet.devnet_kdd19 import fit_devnet

                    baseline_model = fit_devnet(
                        random_state=self.random_seed,
                        x=this_data.train_alarm[0].reshape(this_data.train_alarm[0].shape[0], -1),
                        y=this_data.train_alarm[1].reshape(this_data.train_alarm[1].shape[0], -1),
                        **kwargs
                    )
            except:
                # DAGMM sometimes has problems on IDS
                print(f"Could not fit {baseline}. Aborting.")
                return

            # Save the baseline method model
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=False)

            if baseline == 'DAGMM':
                baseline_model.save(out_path)
            elif baseline == "DevNet":
                baseline_model.save_weights(str(out_path))
            else:
                dump(baseline_model, out_path)

            self.target_folder = self.parse_name(baseline) if is_subfolder else ""

    def train_baselines(self, baselines: List[str]):
        """
        Train multiple baselines
        :param baselines: list of baseline methods
        :return:
        """
        for cur_baseline in baselines:
            self.train_baseline(baseline=cur_baseline)

    # -- Helpers --
    @staticmethod
    def thresh_middle(vals_norm: np.ndarray, vals_anom: np.ndarray, std_shift: int = 0):
        """
        Calculate a detection threshold based on the mean and std of values
        :param vals_norm: normal samples
        :param vals_anom: anomalous samples
        :param std_shift: by how many std should the threshold be shifted
        :return:
        """

        # Usually, anomalous samples should have a higher reconstruction error, but we better check
        vals_low = vals_norm if vals_norm.mean() < vals_anom.mean() else vals_anom
        vals_high = vals_anom if vals_norm.mean() < vals_anom.mean() else vals_norm
        if vals_low is not vals_norm or vals_high is not vals_anom:
            warn("Anomalous samples have a smaller reconstruction threshold than normal ones.")

        # Add std if desired
        mean_low = vals_low.mean() + std_shift * vals_low.std()
        mean_high = vals_high.mean() - std_shift * vals_high.std()

        # Take the middle as threshold
        mean_thresh = (mean_high - mean_low) / 2.

        return mean_thresh

    @staticmethod
    def automatic_class_weights(train_data: tuple, n_anomalies: int, max_weight: int = 100) -> int:
        """
        Weight anomaly classes more than normal samples to account for the class imbalances
        :param train_data: training data (x, y)
        :param n_anomalies: number of anomalies in the sample
        :param max_weight: cap the maximum weight returned
        """
        anom_weight = math.ceil(len(train_data[1][train_data[1] == 0]) / n_anomalies)

        # We should be sure that we return something useful
        assert anom_weight >= 1

        if max_weight:
            anom_weight = min(anom_weight, max_weight)

        return anom_weight

    @staticmethod
    def thresh_pred(in_pred: np.ndarray, thresh_pred: float = .5):
        """
        Make a binary prediction based on a threshold
        :param in_pred: original prediction with uncertainties
        :param thresh_pred: prediction threshold
        :return: binary prediction
        """
        out_pred = in_pred.copy()
        out_pred[in_pred >= thresh_pred] = 1
        out_pred[in_pred < thresh_pred] = 0

        return out_pred

    @staticmethod
    def parse_name(in_conf: ExperimentConfig, additional_info: str = "", remove_after: str = None) -> str:
        """
        Convert configuration to a nicer file name
        :param in_conf: dictionary
        :param additional_info: a string that will be appended to the name
        :param remove_after: remove everything after andincluding the given keyword
        :return: string describing the dictionary
        """
        out_str = str(in_conf)

        # Remove some parts
        if remove_after:
            out_str = out_str.partition(remove_after)[0]

        # Remove full stops as otherwise the path may be strange
        out_str = re.sub(r"[\\'.<>\[\]()\s]", "", out_str)

        out_str += additional_info

        return out_str

    @staticmethod
    def dict_to_str(in_dict: dict) -> str:
        """
        Parse the values of a dictionary as string
        :param in_dict: dictionary
        :return: dictionary with the same keys but the values as string
        """
        out_dict = {cur_key: str(cur_val) for cur_key, cur_val in in_dict.items()}

        return out_dict

    def get_model_path(
            self, base_path: Path,
            file_name: str = None, file_suffix: str = ".h5",
            sub_folder: str = "", sub_sub_folder: str = "",
    ) -> Path:
        """
        Get the path to save the NN models
        :param base_path: path to the project
        :param file_name: name of the model file (prefix is prepended)
        :param file_suffix: suffix of the file
        :param sub_folder: folder below model folder, e.g. for alarm/target
        :param sub_sub_folder: folder below subfolder, e.g. architecture details
        :return:
        """
        out_path = base_path / "models"

        if sub_folder:
            out_path /= sub_folder

        if sub_sub_folder:
            out_path /= sub_sub_folder

        if file_name:
            out_path /= f"{self.save_prefix}_{file_name}"
            out_path = out_path.with_suffix(file_suffix)

        return out_path


# Calculate metrics at the end of each epoch
# based on https://github.com/keras-team/keras/issues/5794#issuecomment-303683985
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        super(Metrics, self).__init__()

        self.training_data = training_data
        # For some reason the validation data cannot be accessed otherwise
        self.validation_data = validation_data

    def on_epoch_end(self, batch, logs={}):
        # Predict and make binary
        train_pred = self.model.predict(self.training_data[0])
        train_pred = ExperimentWrapper.thresh_pred(train_pred)
        val_pred = self.model.predict(self.validation_data[0])
        val_pred = ExperimentWrapper.thresh_pred(val_pred)

        # Calculate metrics
        train_f1 = f1_score(self.training_data[1], train_pred)
        train_prec = precision_score(self.training_data[1], train_pred)
        train__rec = recall_score(self.training_data[1], train_pred)
        val_f1 = f1_score(self.validation_data[1], val_pred)
        val_prec = precision_score(self.validation_data[1], val_pred)
        val_rec = recall_score(self.validation_data[1], val_pred)

        print(f"Training scores     F1: {train_f1}, Precision: {train_prec}, Recall: {train__rec}")
        print(f"Validation scores   F1: {val_f1}, Precision: {val_prec}, Recall: {val_rec}")
