import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot as plt

from libs.A3 import A3
from libs.DataHandler import MNIST
from libs.architecture import conv_ae, alarm_net, VariationalAutoEncoder, RandomNoise
from libs.ExperimentWrapper import ExperimentWrapper
from utils import BASE_PATH

NORMAL_CLASSES = [0, 1, 2, 3, 4, 5]
ANOMALY_CLASSES = [6, 7, 8, 9]
ALL_CLASSES = NORMAL_CLASSES.copy()
ALL_CLASSES.extend(ANOMALY_CLASSES)

PARAMETERS = {
    "anomaly_layer_dims": [
        []
    ],
    "alarm_layer_dims": [
        [1000, 500, 200, 75]
    ],
    "in_l1": [
        0.0, 0.1
    ],
    "in_l2": [
        0.0, 0.1
    ],
    "out_l1": [
        0.0, 0.1
    ],
    "out_l2": [
        0.0, 0.1
    ],
    "anomaly_weight": [
        0.1, 1.0,
    ],
    "anomaly_var": [
        5.0
    ],
}

if __name__ == "__main__":
    # Open data
    mnist = MNIST(random_state=2409)
    train_val = "train"
    train_target = mnist.get_target_autoencoder_data(data_split=train_val, include_classes=NORMAL_CLASSES)
    train_alarm = mnist.get_alarm_data(
        data_split=train_val, include_classes=NORMAL_CLASSES, anomaly_classes=ANOMALY_CLASSES,
        n_anomaly_samples=0
    )
    train_anomaly = np.ones_like(train_alarm[1])

    val_target = mnist.get_target_autoencoder_data(data_split="val", include_classes=NORMAL_CLASSES)
    val_alarm = mnist.get_alarm_data(data_split="val", include_classes=ALL_CLASSES, anomaly_classes=ANOMALY_CLASSES)
    val_anomaly = np.ones_like(val_alarm[1])

    test_alarm = mnist.get_alarm_data(data_split="test", include_classes=ALL_CLASSES, anomaly_classes=ANOMALY_CLASSES)

    # Train a model for each configuration
    for cur_config in ParameterGrid(PARAMETERS):

        print(f"Currently evaluating {cur_config}")

        # Create anomaly network
        if cur_config["anomaly_layer_dims"]:
            model_anomaly = VariationalAutoEncoder(
                input_shape=mnist.shape,
                layer_dims=cur_config["anomaly_layer_dims"],
                anomaly_var=cur_config["anomaly_var"]
            )
            model_anomaly.compile(optimizer=tf.keras.optimizers.Adam(.001))
            model_anomaly.fit(
                train_target[0],
                validation_data=(val_target[0], None),
                epochs=30, batch_size=256
            )
        else:
            model_anomaly = RandomNoise()

        # Create target network
        model_target = conv_ae(input_shape=mnist.shape)
        model_target.compile(optimizer='adam', loss='binary_crossentropy')
        model_target.fit(
            train_target[0], train_target[1],
            validation_data=val_target,
            epochs=30, batch_size=256
        )

        # Create alarm and overall network
        model_a3 = A3(
            target_network=model_target,
            anomaly_network=model_anomaly,
            anomaly_loss_weight=cur_config["anomaly_weight"]
        )
        model_alarm = alarm_net(
            layer_dims=cur_config["alarm_layer_dims"],
            input_shape=model_a3.get_alarm_shape(),
            in_l1=cur_config["in_l1"],
            in_l2=cur_config["in_l2"],
            out_l1=cur_config["out_l1"],
            out_l2=cur_config["out_l2"],
        )
        model_a3.add_alarm_network(model_alarm)

        model_a3.compile(
            optimizer=tf.keras.optimizers.Adam(.00001),
            loss="binary_crossentropy",
        )
        model_a3.fit(
            [train_alarm[0]],
            [train_alarm[1]],
            validation_data=(
                [val_alarm[0]],
                [val_alarm[1]]
            ),
            epochs=60, batch_size=256, verbose=1,
        )

        # Predict
        pred_y = model_a3.predict([val_alarm[0]], get_activation=True)
        pred_y = pred_y if not isinstance(pred_y, list) else pred_y[0]

        # Plot ROC
        fpr_a3, tpr_a3, thresholds_a3 = roc_curve(
            y_true=val_alarm[1], y_score=pred_y
        )
        plt.plot(fpr_a3, tpr_a3, label=f"A3")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(
            (BASE_PATH / "results" / "parameters" / "mnist" / ExperimentWrapper.parse_name(cur_config)).with_suffix(".png")
        )
        plt.clf()

