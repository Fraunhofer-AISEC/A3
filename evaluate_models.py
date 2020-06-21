import argparse

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from pathlib import Path
from joblib import load

from baselines.dagmm.dagmm import DAGMM
from baselines.devnet.devnet_kdd19 import predict_devnet

from libs.ExperimentWrapper import ExperimentConfig, ExperimentWrapper
from libs.DataHandler import MNIST, CreditCard, NSL_KDD, IDS
from libs.A3 import A3
from libs.Metrics import evaluate, evaluate_multiple
from libs.architecture import RandomNoise, VariationalAutoEncoder
from utils import BASE_PATH, N_ANOMALY_SAMPLES

# Saves the ROC and the metrics evaluated on the test data for all experiments specified in the configs below.
# Expects the model path to be 'BASE_PATH / "mnist_SEED" / "models" / "a3"' respectively.
# On a previous version, we used threshold-dependent metrics (e.g. precision). The threshold was determined on the ROC
# using the validation data. We switched to the AUC and the AP for the recent version of the paper. Please ignore the
# old metrics as they are now determined based on the test data which causes overoptimistic results.

def roc_to_threshold(tpr: np.ndarray, fpr: np.ndarray, thresholds: np.ndarray, max_fpr: float = .05) -> float:
    """
    Return the threshold that causes the highest tpr for the given maximum fpr
    :param tpr: true positive rate
    :param fpr: false positive rate
    :param thresholds: threshold at the given tpr/fpr
    :param max_fpr: maximum allowed fpr
    :return: threshold with highest tpr at the maximum fpr
    """
    # Find the index where fpr is below the maximum
    try:
        idx_fpr_max = np.argmax(fpr[fpr < max_fpr])
    except ValueError:
        print("No best FPR found!")
        idx_fpr_max = 0

    # Find maximum tpr for this index
    try:
        idx_tpr_max = np.argmax(tpr[:idx_fpr_max])
    except ValueError:
        print("No best TPR found!")
        idx_tpr_max = 0

    # Find values at this index
    tpr_max = tpr[idx_tpr_max]
    fpr_max = fpr[idx_tpr_max]
    thresh_max = thresholds[idx_tpr_max]

    return thresh_max


def roc_to_pandas(fpr: np.ndarray, tpr: np.ndarray, suffix: str, decimals: int = 3) -> pd.DataFrame:
    """
    Round the ROC results to save some computation time in TikZ (in fact, the IDS results are too big otherwise)
    :param fpr: false positive rate
    :param tpr: true positive rate
    :param suffix: string appended to the column names
    :param decimals: decimals kept
    :return: DataFrame with the rounded TPR&FPR values
    """

    out_df = pd.concat([
        pd.Series(fpr, name=f"fpr_{suffix}"),
        pd.Series(tpr, name=f"tpr_{suffix}")
    ], axis=1)

    # Round and delete duplicates (look for duplicates in the FPR)
    out_df = out_df.round(decimals=decimals)
    out_df = out_df.drop_duplicates(subset=f"fpr_{suffix}", ignore_index=True)

    return out_df


if __name__ == '__main__':
    # Configuration
    this_parse = argparse.ArgumentParser(description="Evaluate A^3 performance on all experiments")
    this_parse.add_argument(
        "random_seed", type=int, help="Seed to fix randomness"
    )
    this_parse.add_argument(
        "--folder_suffix", default="", type=str, help="Suffix added to the foldernames (e.g., the random seed)"
    )
    this_parse.add_argument(
        "--in_path", default=BASE_PATH / "models", type=Path, help="Base input path for the models"
    )
    this_parse.add_argument(
        "--out_path", default=BASE_PATH / "results", type=Path, help="Base output path for the results"
    )
    this_parse.add_argument(
        "--use_vae", default=False, type=bool, help="Use a VAE as anomaly network instead of noise (experiment 4)"
    )
    this_args = this_parse.parse_args()

    # Config
    RANDOM_SEED = this_args.random_seed
    MAX_FPR = [0.000001, 0.00001, 0.0001, 0.001, 0.01]

    # Data path
    OUT_PATH = this_args.out_path
    FOLDER_SUFFIX = this_args.folder_suffix
    BASE_PATH = this_args.in_path

    # Take the right setting
    if this_args.use_vae:
        # For the VAE, we only consider 0 anomaly samples
        MODEL_N_ANOMALIES = [0]

        MODEL_CONFIG = [
            {
                "conf": ExperimentConfig(MNIST(random_state=RANDOM_SEED), list(range(0, 6)), [6, 7], [6, 7]),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist_vae{FOLDER_SUFFIX}" / "models" / "a3",
                "vae_layers": [800, 400, 100, 25],
                "thresh": True
            },
            {
                "conf": ExperimentConfig(MNIST(random_state=RANDOM_SEED), list(range(0, 6)), [6, 7],
                                         list(range(6, 10))),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist_vae{FOLDER_SUFFIX}" / "models" / "a3",
                "vae_layers": [800, 400, 100, 25],
                "thresh": False
            },
            {
                "conf": ExperimentConfig(MNIST(random_state=RANDOM_SEED), list(range(4, 10)), [0, 1], [0, 1]),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist_vae{FOLDER_SUFFIX}" / "models" / "a3",
                "vae_layers": [800, 400, 100, 25],
                "thresh": True
            },
            {
                "conf": ExperimentConfig(MNIST(random_state=RANDOM_SEED), list(range(4, 10)), [0, 1],
                                         list(range(0, 4))),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist_vae{FOLDER_SUFFIX}" / "models" / "a3",
                "vae_layers": [800, 400, 100, 25],
                "thresh": False
            },
        ]

    else:
        # We'll reverse the order such that we can automatically determine the threshold on the maximum available anomalies
        MODEL_N_ANOMALIES = list(reversed(sorted(N_ANOMALY_SAMPLES)))

        this_ids = IDS(random_state=RANDOM_SEED)
        MODEL_CONFIG = [
            # MNIST experiments
            {
                "conf": ExperimentConfig(MNIST(random_state=RANDOM_SEED), list(range(0, 6)), [6, 7], [6, 7]),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": True
            },
            {
                "conf": ExperimentConfig(MNIST(random_state=RANDOM_SEED), list(range(0, 6)), [6, 7],
                                         list(range(6, 10))),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": False
            },
            {
                "conf": ExperimentConfig(MNIST(random_state=RANDOM_SEED), list(range(4, 10)), [0, 1], [0, 1]),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": True
            },
            {
                "conf": ExperimentConfig(MNIST(random_state=RANDOM_SEED), list(range(4, 10)), [0, 1],
                                         list(range(0, 4))),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": False
            },
            # CC Experiments
            {
                "conf": ExperimentConfig(CreditCard(random_state=RANDOM_SEED), [0], [1], [1]),
                "prefix": f"CC_{RANDOM_SEED}_",
                "path": BASE_PATH / f"creditcard{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": True
            },
            # KDD experiments
            {
                "conf": ExperimentConfig(NSL_KDD(
                    random_state=RANDOM_SEED), ["normal"], ["DoS", "Probe"],
                    ["DoS", "Probe"]
                ),
                "prefix": f"NSL_KDD_{RANDOM_SEED}_",
                "path": BASE_PATH / f"nsl_kdd{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": True
            },
            {
                "conf": ExperimentConfig(NSL_KDD(
                    random_state=RANDOM_SEED), ["normal"], ["DoS", "Probe"],
                    ["DoS", "Probe", "R2L", "U2R"]
                ),
                "prefix": f"NSL_KDD_{RANDOM_SEED}_",
                "path": BASE_PATH / f"nsl_kdd{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": False
            },
            {
                "conf": ExperimentConfig(NSL_KDD(
                    random_state=RANDOM_SEED), ["normal"], ["R2L", "U2R"],
                    ["R2L", "U2R"]
                ),
                "prefix": f"NSL_KDD_{RANDOM_SEED}_",
                "path": BASE_PATH / f"nsl_kdd{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": True
            },
            {
                "conf": ExperimentConfig(NSL_KDD(
                    random_state=RANDOM_SEED), ["normal"], ["R2L", "U2R"],
                    ["DoS", "Probe", "R2L", "U2R"]
                ),
                "prefix": f"NSL_KDD_{RANDOM_SEED}_",
                "path": BASE_PATH / f"nsl_kdd{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": False
            },
            # EMNIST experiments
            {
                "conf": ExperimentConfig(MNIST(
                    random_state=RANDOM_SEED, enrich_mnist_by=[10, 11, 12, 13, 14, 31, 32, 33, 34, 35],
                ), list(range(0, 10)), [10, 11, 12, 13, 14], [10, 11, 12, 13, 14]),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist_emnist{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": True
            },
            {
                "conf": ExperimentConfig(MNIST(
                    random_state=RANDOM_SEED, enrich_mnist_by=[10, 11, 12, 13, 14, 31, 32, 33, 34, 35],
                ), list(range(0, 10)), [10, 11, 12, 13, 14], [10, 11, 12, 13, 14, 31, 32, 33, 34, 35]),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist_emnist{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": False
            },
            {
                "conf": ExperimentConfig(MNIST(
                    random_state=RANDOM_SEED, enrich_mnist_by=[10, 11, 12, 13, 14, 31, 32, 33, 34, 35],
                ), list(range(0, 10)), [31, 32, 33, 34, 35], [31, 32, 33, 34, 35]),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist_emnist{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": True
            },
            {
                "conf": ExperimentConfig(MNIST(
                    random_state=RANDOM_SEED, enrich_mnist_by=[10, 11, 12, 13, 14, 31, 32, 33, 34, 35],
                ), list(range(0, 10)), [31, 32, 33, 34, 35], [10, 11, 12, 13, 14, 31, 32, 33, 34, 35]),
                "prefix": f"MNIST_{RANDOM_SEED}_",
                "path": BASE_PATH / f"mnist_emnist{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": False
            },
            # IDS experiments
            {
                "conf": ExperimentConfig(this_ids, ["Benign"], ["BruteForce", "DoS", "WebAttacks", "Infiltration"],
                                         ["BruteForce", "DoS", "WebAttacks", "Infiltration"]
                                         ),
                "prefix": f"IDS_{RANDOM_SEED}_",
                "path": BASE_PATH / f"ids{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": True
            },
            {
                "conf": ExperimentConfig(this_ids, ["Benign"], ["BruteForce", "DoS", "WebAttacks", "Infiltration"],
                                         ["BruteForce", "DoS", "WebAttacks", "Infiltration", "Bot"]
                                         ),
                "prefix": f"IDS_{RANDOM_SEED}_",
                "path": BASE_PATH / f"ids{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": False
            },
            {
                "conf": ExperimentConfig(this_ids, ["Benign"], ["Bot", "Infiltration", "WebAttacks", "DoS"],
                                         ["Bot", "Infiltration", "WebAttacks", "DoS"]
                                         ),
                "prefix": f"IDS_{RANDOM_SEED}_",
                "path": BASE_PATH / f"ids{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": True
            },
            {
                "conf": ExperimentConfig(this_ids, ["Benign"], ["Bot", "Infiltration", "WebAttacks", "DoS"],
                                         ["Bot", "Infiltration", "WebAttacks", "DoS", "BruteForce"]
                                         ),
                "prefix": f"IDS_{RANDOM_SEED}_",
                "path": BASE_PATH / f"ids{FOLDER_SUFFIX}" / "models" / "a3",
                "thresh": False
            },
        ]

    # We must determine a threshold on the first entry, otherwise we get some Nones
    assert MODEL_CONFIG[0]["thresh"] is True

    # As we loop through all data sets, we might as well evaluate them
    column_names = ["AUC-ROC", "AUC-PR"]
    for cur_fpr in MAX_FPR:
        column_names.extend([f"F1_{cur_fpr}", f"Precision_{cur_fpr}", f"Recall_{cur_fpr}"])
    all_results = pd.DataFrame(columns=column_names)

    for cur_conf in MODEL_CONFIG:

        # Due to DAGMM (not TF2 compatible), we have to reset tensorflow after each iteration
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_v2_behavior()

        print(f"Currently evaluating {ExperimentWrapper.parse_name(cur_conf['conf'])}")

        # Get output name
        out_path = OUT_PATH / f"{cur_conf['prefix']}{ExperimentWrapper.parse_name(cur_conf['conf'])}"
        # # Check if exists
        # if out_path.with_suffix(".csv").exists():
        #     print("This has already been evaluated. Please delete the old output first.")
        #     continue

        # Prepare ROC plot and data output
        fig = plt.figure()
        # We need an x and y column for all n_anomalies as well as the AE
        dat = pd.DataFrame()

        # Request validation (for the threshold) and test data (for the evaluation)
        this_data = cur_conf["conf"].to_data()

        # Determine the anomaly network
        if "vae_layers" in cur_conf:
            anomaly_net = VariationalAutoEncoder(
                input_shape=this_data.data_shape,
                layer_dims=cur_conf["vae_layers"]
            )
        else:
            anomaly_net = RandomNoise()

        # Combine to overall network
        this_a3 = A3(
            anomaly_network=anomaly_net
        )

        # Store the threshold if we use the model to determine the threshold
        if cur_conf["thresh"]:
            cur_thresh_a3 = None
            cur_thresh_ae = None
            cur_thresh_if = None
            cur_thresh_dagmm = None
            cur_thresh_sad = None
            cur_thresh_dev = None

        # == A3 ==
        for cur_n_anomaly in MODEL_N_ANOMALIES:
            this_prefix = ExperimentWrapper.parse_name(cur_conf["conf"], additional_info=f"anom={cur_n_anomaly}")
            this_a3.load_all(cur_conf["path"], prefix=cur_conf["prefix"] + this_prefix)

            # Predict A^3 labels
            val_a3 = this_a3.predict(x=this_data.val_alarm[0])
            test_a3 = this_a3.predict(x=this_data.test_alarm[0])

            # Plot ROC
            fpr_a3, tpr_a3, thresholds_a3 = roc_curve(
                y_true=this_data.test_alarm[1], y_score=test_a3
            )
            plt.plot(fpr_a3, tpr_a3, label=f"A3 n_anomaly={cur_n_anomaly}")
            # Save ROC
            cur_dat = roc_to_pandas(fpr=fpr_a3, tpr=tpr_a3, suffix=f"{cur_n_anomaly}")
            dat = pd.concat([dat, cur_dat], axis=1)

            # Get best threshold
            # NOTE: before we took threshold-independent metrics (e.g. the AUC), we took the val data here
            thresh_max_a3 = [
                roc_to_threshold(tpr=tpr_a3, fpr=fpr_a3, thresholds=thresholds_a3, max_fpr=cur_fpr)
                for cur_fpr in MAX_FPR
            ]
            print(f"For {cur_n_anomaly} anomalies, the best threshold is at {thresh_max_a3}.")

            # Use the current threshold if desired
            if not cur_thresh_a3:
                print("We'll use the A3 threshold for the subsequent A3 evaluation.")
                cur_thresh_a3 = thresh_max_a3

            # We'll also do the evaluation on the test data
            all_results.loc[
                f"{ExperimentWrapper.parse_name(cur_conf['conf'])}_{cur_n_anomaly}", :
            ] = evaluate_multiple(a3=this_a3, test_alarm=this_data.test_alarm, thresholds=MAX_FPR)

        # == Autoencoder == (if there is an autoencoder)
        try:
            # Target should always be the same, so predict with the last open target
            val_ae = this_a3.target_network.predict(x=this_data.val_alarm[0])
            test_ae = this_a3.target_network.predict(x=this_data.test_alarm[0])
            # Get all reconstruction errors
            val_ae = np.square(val_ae - this_data.val_alarm[0])
            test_ae = np.square(test_ae - this_data.test_alarm[0])
            # Collaps to one dimension per sample
            val_ae = np.reshape(val_ae, (val_ae.shape[0], -1))
            test_ae = np.reshape(test_ae, (test_ae.shape[0], -1))
            # Our threshold is based on the validation MSE
            val_ae = np.mean(val_ae, axis=1)
            test_ae = np.mean(test_ae, axis=1)
            val_ae = np.reshape(val_ae, (-1, 1))
            test_ae = np.reshape(test_ae, (-1, 1))

            # Get FPR/TPR data
            fpr_ae, tpr_ae, thresholds_ae = roc_curve(y_true=this_data.test_alarm[1], y_score=test_ae)
            # Save ROC
            cur_dat = roc_to_pandas(fpr=fpr_ae, tpr=tpr_ae, suffix=f"ae")
            dat = pd.concat([dat, cur_dat], axis=1)
            plt.plot(fpr_ae, tpr_ae, label="Autoencoder")

            # Get best threshold
            thresh_max_ae = [
                roc_to_threshold(tpr=tpr_ae, fpr=fpr_ae, thresholds=thresholds_ae, max_fpr=cur_fpr)
                for cur_fpr in MAX_FPR
            ]
            print(f"For the autoencoder, the best threshold is at {thresh_max_ae}.")

            # Use the current threshold if desired
            if not cur_thresh_ae:
                print("We'll use the AE threshold for the subsequent AE evaluation.")
                cur_thresh_ae = thresh_max_ae

            # Evaluate
            all_results.loc[
                f"{ExperimentWrapper.parse_name(cur_conf['conf'])}_ae", :
            ] = evaluate_multiple(a3=test_ae, test_alarm=this_data.test_alarm, thresholds=MAX_FPR)

        except ValueError:
            # If we look at a classifier, we don't have any baseline method
            pass

        # == Isolation Forest ==
        this_prefix = ExperimentWrapper.parse_name(cur_conf["conf"])
        this_prefix = cur_conf["prefix"] + this_prefix
        this_forest = load((cur_conf["path"].parent / "IsolationForest" / this_prefix).with_suffix(".joblib"))

        # Predict anomaly score
        val_if = this_forest.decision_function(
            this_data.val_alarm[0].reshape((this_data.val_alarm[0].shape[0], -1))
        )
        test_if = this_forest.decision_function(
            this_data.test_alarm[0].reshape((this_data.test_alarm[0].shape[0], -1))
        )
        # We need to invert the results as "The lower, the more abnormal."
        # See also https://github.com/scikit-learn/scikit-learn/blob/master/benchmarks/bench_isolation_forest.py
        val_if *= -1
        test_if *= -1

        # Plot ROC
        fpr_if, tpr_if, thresholds_if = roc_curve(
            y_true=this_data.test_alarm[1], y_score=test_if
        )
        plt.plot(fpr_if, tpr_if, label=f"Isolation Forest")
        # Save ROC
        cur_dat = roc_to_pandas(fpr=fpr_if, tpr=tpr_if, suffix="if")
        dat = pd.concat([dat, cur_dat], axis=1)

        # Get best threshold
        thresh_max_if = [
            roc_to_threshold(tpr=tpr_if, fpr=fpr_if, thresholds=thresholds_if, max_fpr=cur_fpr)
            for cur_fpr in MAX_FPR
        ]
        print(f"For Isolation Forest, the best threshold is at {thresh_max_if}.")

        # Use the current threshold if desired
        if not cur_thresh_if:
            print("We'll use the IF threshold for the subsequent IF evaluation.")
            cur_thresh_if = thresh_max_if

        # We'll also do the evaluation on the test data
        all_results.loc[
            f"{ExperimentWrapper.parse_name(cur_conf['conf'])}_if", :
        ] = evaluate_multiple(a3=test_if, test_alarm=this_data.test_alarm, thresholds=MAX_FPR)

        # == DAGMM ==
        try:
            this_prefix = ExperimentWrapper.parse_name(cur_conf["conf"])
            this_prefix = cur_conf["prefix"] + this_prefix

            # Load model
            this_dagmm = DAGMM(
                comp_hiddens=[60, 30, 10, 1], comp_activation=tf.nn.tanh,
                est_hiddens=[10, 4], est_dropout_ratio=0.5, est_activation=tf.nn.tanh,
                learning_rate=0.0001, epoch_size=200, minibatch_size=1024,
                random_seed=RANDOM_SEED
            )
            this_dagmm.restore((cur_conf["path"].parent / "DAGMM" / this_prefix))

            # Predict anomaly score
            val_dagmm = this_dagmm.predict(
                this_data.val_alarm[0].reshape((this_data.val_alarm[0].shape[0], -1))
            )
            test_dagmm = this_dagmm.predict(
                this_data.test_alarm[0].reshape((this_data.test_alarm[0].shape[0], -1))
            )

            # Plot ROC
            fpr_dagmm, tpr_dagmm, thresholds_dagmm = roc_curve(
                y_true=this_data.test_alarm[1], y_score=test_dagmm
            )
            plt.plot(fpr_dagmm, tpr_dagmm, label=f"DAGMM")
            # Save ROC
            cur_dat = roc_to_pandas(fpr=fpr_dagmm, tpr=tpr_dagmm, suffix="dagmm")
            dat = pd.concat([dat, cur_dat], axis=1)

            # Get best threshold
            thresh_max_dagmm = [
                roc_to_threshold(tpr=tpr_dagmm, fpr=fpr_dagmm, thresholds=thresholds_dagmm, max_fpr=cur_fpr)
                for cur_fpr in MAX_FPR
            ]
            print(f"For DAGMM, the best threshold is at {thresh_max_dagmm}.")

            # Use the current threshold if desired
            if not cur_thresh_dagmm:
                print("We'll use the DAGMM threshold for the subsequent DAGMM evaluation.")
                cur_thresh_dagmm = thresh_max_dagmm

            # We'll also do the evaluation on the test data
            all_results.loc[
                f"{ExperimentWrapper.parse_name(cur_conf['conf'])}_dagmm", :
            ] = evaluate_multiple(a3=test_dagmm, test_alarm=this_data.test_alarm, thresholds=MAX_FPR)
        except Exception:
            print("No DAGMM model found!")

        # == DevNet ==
        try:
            this_prefix = ExperimentWrapper.parse_name(cur_conf["conf"])
            this_prefix = cur_conf["prefix"] + this_prefix

            val_dev = predict_devnet(
                model_name=str((cur_conf["path"].parent / "DevNet" / this_prefix).with_suffix(".h5")),
                x=this_data.val_alarm[0].reshape(this_data.val_alarm[0].shape[0], -1)
            )
            test_dev = predict_devnet(
                model_name=str((cur_conf["path"].parent / "DevNet" / this_prefix).with_suffix(".h5")),
                x=this_data.test_alarm[0].reshape(this_data.test_alarm[0].shape[0], -1)
            )

            # Plot ROC
            fpr_dev, tpr_dev, thresholds_dev = roc_curve(
                y_true=this_data.test_alarm[1], y_score=test_dev
            )
            plt.plot(fpr_dev, tpr_dev, label=f"DevNet")
            # Save ROC
            cur_dat = roc_to_pandas(fpr=fpr_dev, tpr=tpr_dev, suffix="devnet")
            dat = pd.concat([dat, cur_dat], axis=1)

            # Get best threshold
            thresh_max_dev = [
                roc_to_threshold(tpr=tpr_dev, fpr=fpr_dev, thresholds=thresholds_dev, max_fpr=cur_thresh)
                for cur_thresh in MAX_FPR
            ]
            print(f"For DevNet, the best threshold is at {thresh_max_dev}.")

            # Use the current threshold if desired
            if not cur_thresh_dev:
                print("We'll use the DevNet threshold for the subsequent DevNet evaluation.")
                cur_thresh_dev = thresh_max_dev

            # We'll also do the evaluation on the test data
            all_results.loc[
                f"{ExperimentWrapper.parse_name(cur_conf['conf'])}_devnet", :
            ] = evaluate_multiple(a3=test_dev, test_alarm=this_data.test_alarm, thresholds=MAX_FPR)

        except (FileNotFoundError, OSError):
            print("No DevNet model found! Ignoring.")

        # Save data
        dat.to_csv(out_path.with_suffix(".csv"), index=False)

        # Plot ROC curve
        plt.plot([0, 1], [0, 1], label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        # plt.show()
        plt.savefig(out_path.with_suffix(".png"))

    # Loop done? Save all results
    all_results.to_csv((OUT_PATH / f"all_results_{RANDOM_SEED}").with_suffix(".csv"))
    pass
