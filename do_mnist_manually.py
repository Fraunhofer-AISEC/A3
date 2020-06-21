import numpy as np
import tensorflow as tf

from libs.A3 import A3
from libs.DataHandler import MNIST
from libs.architecture import conv_ae, alarm_net, VariationalAutoEncoder, RandomNoise
from utils import BASE_PATH

TRAIN_TARGET = True
NORMAL_CLASSES = [0, 1, 2]
ANOMALY_CLASSES = [3, 4]
ALL_CLASSES = NORMAL_CLASSES.extend(ANOMALY_CLASSES)

if __name__ == "__main__":
    # Open data
    mnist = MNIST(random_state=2409)
    train_val = "train"
    train_target = mnist.get_target_autoencoder_data(data_split=train_val, include_classes=NORMAL_CLASSES)
    train_alarm = mnist.get_alarm_data(
        data_split=train_val, include_classes=NORMAL_CLASSES, anomaly_classes=ANOMALY_CLASSES,
        n_anomaly_samples=1
    )

    val_target = mnist.get_target_autoencoder_data(data_split="val", include_classes=NORMAL_CLASSES)
    val_alarm = mnist.get_alarm_data(data_split="val", include_classes=ALL_CLASSES, anomaly_classes=ANOMALY_CLASSES)
    val_anomaly = np.ones_like(val_alarm[1])

    test_alarm = mnist.get_alarm_data(data_split="test", include_classes=ALL_CLASSES, anomaly_classes=ANOMALY_CLASSES)

    # Create anomaly network
    random_noise = RandomNoise("normal")
    model_vae = VariationalAutoEncoder(
        input_shape=mnist.shape,
        layer_dims=[800, 400, 100, 25]
    )
    model_vae.compile(optimizer=tf.keras.optimizers.Adam(.001))
    # Subclassed Keras models don't know about the shapes in advance... build() didn't do the trick
    model_vae.fit(train_target[0], epochs=0, batch_size=256)

    if TRAIN_TARGET:
        # Keep fitting the anomaly network
        # model_vae.fit(
        #     train_target[0],
        #     validation_data=(val_target[0], None),
        #     epochs=15, batch_size=256
        # )

        # Create target network
        model_target = conv_ae(input_shape=mnist.shape)
        model_target.compile(optimizer='adam', loss='binary_crossentropy')
        model_target.fit(
            train_target[0], train_target[1],
            validation_data=val_target,
            epochs=15, batch_size=256
        )

        # Create alarm and overall network
        model_a3 = A3(target_network=model_target)
        model_a3.add_anomaly_network(random_noise)
        model_alarm = alarm_net(
            layer_dims=[1000, 500, 200, 75],
            input_shape=model_a3.get_alarm_shape(),
        )
        model_a3.add_alarm_network(model_alarm)
    else:
        model_a3 = A3()
        model_a3.load_all(
            anomaly_model=random_noise,
            basepath=BASE_PATH / "models", prefix="mnist"
        )

    model_a3.compile(
        optimizer=tf.keras.optimizers.Adam(.00001),
        loss="binary_crossentropy",
    )
    model_a3.fit(
        train_alarm[0],
        train_alarm[1],
        validation_data=val_alarm,
        epochs=30, batch_size=256, verbose=1,
    )
    model_a3.save(basepath=BASE_PATH / "models", prefix="mnist")
