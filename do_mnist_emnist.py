import argparse
import tensorflow as tf

from pathlib import Path

from libs.ExperimentWrapper import ExperimentWrapper, ExperimentConfig
from libs.DataHandler import MNIST
from libs.architecture import conv_ae, dense_ae, alarm_net, RandomNoise, VariationalAutoEncoder
from libs.Metrics import evaluate

from utils import BASE_PATH, N_ANOMALY_SAMPLES

# Configuration
this_parse = argparse.ArgumentParser(description="Train A^3 on all experiments")
this_parse.add_argument(
    "random_seed", type=int, help="Seed to fix randomness"
)
this_parse.add_argument(
    "test_type", type=str, help="Use validation or test data set for the evaluation (not yet integrated)"
)
this_parse.add_argument(
    "target_path", type=Path, help="Location of the target network (here: a classifier)"
)
this_parse.add_argument(
    "--out_path", default=BASE_PATH, type=Path, help="Base output path for the models"
)
this_args = this_parse.parse_args()

# Static configuration
experiment_config_convolutional = [
    # Known anomalies
    ExperimentConfig(MNIST(
        random_state=this_args.random_seed, enrich_mnist_by=[10, 11, 12, 13, 14, 31, 32, 33, 34, 35],), 
        list(range(0, 10)), [10, 11, 12, 13, 14], [10, 11, 12, 13, 14]),
    # Unknown anomalies
    ExperimentConfig(MNIST(
        random_state=this_args.random_seed, enrich_mnist_by=[10, 11, 12, 13, 14, 31, 32, 33, 34, 35],), 
        list(range(0, 10)), [10, 11, 12, 13, 14], [10, 11, 12, 13, 14, 31, 32, 33, 34, 35]),
    # Known anomalies
    ExperimentConfig(MNIST(
        random_state=this_args.random_seed, enrich_mnist_by=[10, 11, 12, 13, 14, 31, 32, 33, 34, 35],), 
        list(range(0, 10)), [31, 32, 33, 34, 35], [31, 32, 33, 34, 35]),
    # Unknown anomalies
    ExperimentConfig(MNIST(
        random_state=this_args.random_seed, enrich_mnist_by=[10, 11, 12, 13, 14, 31, 32, 33, 34, 35]), 
        list(range(0, 10)), [31, 32, 33, 34, 35], [10, 11, 12, 13, 14, 31, 32, 33, 34, 35]),
]


if __name__ == '__main__':

    this_experiment = ExperimentWrapper(
        save_prefix="MNIST",
        data_setup=experiment_config_convolutional, n_anomaly_samples=N_ANOMALY_SAMPLES,
        test_type=this_args.test_type, random_seed=this_args.random_seed, out_path=this_args.out_path
    )

    # Choose anomaly network
    anomaly_net = RandomNoise()

    # Train alarm network
    this_experiment.train_a3(
        alarm_net=alarm_net, anomaly_net=anomaly_net,
        target_path=this_args.target_path,
        network_params={"layer_dims": [1000, 500, 200, 75], "in_l1": 0.0, "in_l2": 0.0, "out_l1": 0.0, "out_l2": 0.0},
        compile_params={"learning_rate": 0.00001, "loss": "binary_crossentropy", "metrics": ["binary_accuracy"]},
        fit_params={"epochs": 60, "batch_size": 256, "verbose": 2},  # note: class weights are automatically adapted
        show_metrics=False, automatic_weights=False,
        is_subfolder=False
    )

    # Train the baselines
    this_experiment.train_baseline(
        baseline="IsolationForest"
    )
    this_experiment.train_baseline(
        baseline="DAGMM",
        comp_hiddens=[60, 30, 10], comp_activation=tf.nn.tanh,
        est_hiddens=[10, 4], est_dropout_ratio=0.5, est_activation=tf.nn.tanh,
        learning_rate=0.0001, epoch_size=200, minibatch_size=1024,
    )
    this_experiment.train_baseline(
        baseline="DevNet"
    )



