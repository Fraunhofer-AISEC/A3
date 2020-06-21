import argparse
import tensorflow as tf

from pathlib import Path

from libs.Metrics import evaluate
from libs.ExperimentWrapper import ExperimentWrapper, ExperimentConfig
from libs.DataHandler import CreditCard
from libs.architecture import conv_ae, dense_ae, alarm_net, RandomNoise, VariationalAutoEncoder
from libs.A3 import A3
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

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
    "--out_path", default=BASE_PATH, type=Path, help="Base output path for the models"
)
this_args = this_parse.parse_args()

# Static configuration
experiment_config_flat = [
    # Known anomalies
    ExperimentConfig(CreditCard(random_state=this_args.random_seed), [0], [1], [1]),
]

if __name__ == '__main__':
    this_experiment = ExperimentWrapper(
        save_prefix="CC",
        data_setup=experiment_config_flat, n_anomaly_samples=N_ANOMALY_SAMPLES,
        test_type=this_args.test_type, random_seed=this_args.random_seed, out_path=this_args.out_path
    )

    # Flat experiments, Train target network
    this_experiment.train_target(
        target_net=dense_ae, network_params={"layer_dims": [50, 25, 10, 5]},
        compile_params={"loss": "binary_crossentropy", "metrics": ["mse", "mae"]},
        fit_params={"epochs": 30, "batch_size": 256, "verbose": 2},
        is_subfolder=False
    )

    # Choose anomaly network
    anomaly_net = RandomNoise()

    # Train alarm network
    this_experiment.train_a3(
        alarm_net=alarm_net, anomaly_net=anomaly_net,
        network_params={"layer_dims": [1000, 500, 200, 75], "in_l1": 0.0, "in_l2": 0.0, "out_l1": 0.0, "out_l2": 0.0},
        compile_params={"learning_rate": 0.00001, "loss": "binary_crossentropy", "metrics": ["binary_accuracy"]},
        fit_params={"epochs": 60, "batch_size": 256, "verbose": 2},
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





