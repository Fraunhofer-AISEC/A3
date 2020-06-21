import argparse
import tensorflow as tf

from pathlib import Path

from libs.ExperimentWrapper import ExperimentWrapper, ExperimentConfig
from libs.DataHandler import MNIST
from libs.architecture import conv_ae, alarm_net, RandomNoise, VariationalAutoEncoder

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
this_parse.add_argument(
    "--use_vae", default=False, type=bool, help="Use a VAE as anomaly network instead of noise (experiment 4)"
)
this_args = this_parse.parse_args()

# We omit all anomaly samples when using the VAE
if this_args.use_vae:
    N_ANOMALY_SAMPLES = [0]

experiment_config_convolutional = [
    # Known anomalies
    ExperimentConfig(MNIST(random_state=this_args.random_seed), list(range(0, 6)), [6, 7], [6, 7]),
    ExperimentConfig(MNIST(random_state=this_args.random_seed), list(range(4, 10)), [0, 1], [0, 1]),
    # Transferability
    ExperimentConfig(MNIST(random_state=this_args.random_seed), list(range(0, 6)), [6, 7], list(range(6, 10))),
    ExperimentConfig(MNIST(random_state=this_args.random_seed), list(range(4, 10)), [0, 1], list(range(0, 4))),
]

if __name__ == '__main__':

    this_experiment = ExperimentWrapper(
        save_prefix="MNIST",
        data_setup=experiment_config_convolutional, n_anomaly_samples=N_ANOMALY_SAMPLES,
        test_type=this_args.test_type, random_seed=this_args.random_seed, out_path=this_args.out_path
    )

    # Convolutional experiments
    this_experiment.train_target(
        target_net=conv_ae,  # note: the architecture is fixed
        compile_params={"loss": "binary_crossentropy", "metrics": ["mse", "mae"]},
        fit_params={"epochs": 30, "batch_size": 256, "verbose": 2},
        is_subfolder=False
    )

    # Choose anomaly network
    if this_args.use_vae:
        anomaly_net = this_experiment.train_anomaly(
            anomaly_net=VariationalAutoEncoder,
            network_params={"layer_dims": [800, 400, 100, 25]},
            compile_params={"loss": "binary_crossentropy", "metrics": ["mse", "mae"]},
            fit_params={"epochs": 30, "batch_size": 256, "verbose": 2},
            is_subfolder=False
        )
    else:
        anomaly_net = RandomNoise()

    # Train alarm network
    this_experiment.train_a3(
        alarm_net=alarm_net, anomaly_net=anomaly_net,
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
