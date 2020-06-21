import abc
import math

import numpy as np
import pandas as pd
import tensorflow as tf

from dataclasses import dataclass
from pathlib import Path

try:
    from emnist import extract_samples
except ModuleNotFoundError:
    pass

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff

from typing import List, Callable, Union, Tuple

from libs.DataTypes import AutoencoderLayers
from utils import BASE_PATH


@dataclass
class DataLabels:
    """
    Class storing test/train data
    """
    # We'll put everything in the train data if no test data was given and split later
    x_train: np.ndarray  # Train data
    y_train: np.ndarray
    x_test: np.ndarray = None  # Test data
    y_test: np.ndarray = None
    x_val: np.ndarray = None  # Validation data
    y_val: np.ndarray = None

    # If needed: a scaler
    scaler: TransformerMixin = None

    # Configuration
    test_split: float = .2  # Test data percentage
    val_split: float = .05  # Validation data percentage
    random_state: int = None  # Random seed

    # Metadata
    shape: tuple = None  # Shape of the data
    available_classes: Union[List[int], List[str]] = None  # all available classes

    ## Class methods
    def __repr__(self):
        return self.__class__.__name__

    ## Retrievers
    def get_target_autoencoder_data(
            self, data_split: str,
            drop_classes: Union[List[int], List[str]] = None, include_classes: Union[List[int], List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for useful for autoencoders
        :param data_split: get data of either "train", "val" or "test"
        :param drop_classes: which classes to drop, drop none if None
        :param include_classes: which classes to include (has priority over drop_classes)
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split)

        # Drop the classes
        if include_classes:
            drop_classes = self.include_to_drop(include_classes)
        this_x = np.delete(this_data[0], np.where(np.isin(this_data[1], drop_classes)), axis=0)

        # For the autoencoder, we don't need much else than x
        return this_x, this_x

    def get_target_classifier_data(
            self, data_split: str,
            drop_classes: Union[List[int], List[str]] = None, include_classes: Union[List[int], List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for useful for classifiers
        :param data_split: get data of either "train", "val" or "test"
        :param drop_classes: which classes to drop, drop none if None
        :param include_classes: which classes to include (has priority over drop_classes)
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split)

        # Drop the classes
        if include_classes:
            drop_classes = self.include_to_drop(include_classes)
        this_x = np.delete(this_data[0], np.where(np.isin(this_data[1], drop_classes)), axis=0)
        this_y = np.delete(this_data[1], np.where(np.isin(this_data[1], drop_classes)), axis=0)

        # Return the data
        return this_x, this_y

    def get_alarm_data(
            self, data_split: str, anomaly_classes: Union[List[int], List[str]], drop_classes: List[int] = None,
            include_classes: List[int] = None,
            n_anomaly_samples: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the labels for the alarm network, i.e. with binary anomaly labels
        :param data_split: get data of either "train", "val" or "test"
        :param anomaly_classes: classes marked as anomaly
        :param drop_classes: which classes to drop (none if None)
        :param include_classes: which classes to include (has priority over drop_classes)
        :param n_anomaly_samples: reduce the number of anomaly samples
        :return: features and labels
        """
        # Get data
        this_data = self._get_data_set(data_split=data_split)

        # Drop the classes
        if include_classes:
            drop_classes = self.include_to_drop(include_classes)
        this_x = np.delete(this_data[0], np.where(np.isin(this_data[1], drop_classes)), axis=0)
        this_y = np.delete(this_data[1], np.where(np.isin(this_data[1], drop_classes)), axis=0)

        # Make labels binary
        this_y[np.where(~np.isin(this_y, anomaly_classes))] = -1
        this_y[np.where(np.isin(this_y, anomaly_classes))] = 0
        this_y += 1
        this_y = this_y.astype("uint8")

        # If desired, reduce the number anomalous samples
        if n_anomaly_samples is not None:
            # IDs of all anomaly samples
            idx_anom = np.where(this_y == 1)[0]

            # Select the indices to delete
            n_delete = len(idx_anom) - n_anomaly_samples
            idx_delete = np.random.choice(idx_anom, size=n_delete, replace=False)

            # Delete indices
            this_x = np.delete(this_x, idx_delete, axis=0)
            this_y = np.delete(this_y, idx_delete, axis=0)

            # Check if we really have the right amount of anomaly samples
            assert np.sum(this_y) == n_anomaly_samples

        return this_x, this_y

    ## Preprocessors
    @abc.abstractmethod
    def _preprocess(self):
        # Preprocessing steps, e.g. data normalisation
        raise NotImplementedError("Implement in subclass")

    def __post_init__(self):
        """
        Process the data
        :return:
        """

        # Fix randomness
        np.random.seed(seed=self.random_state)

        # Get all available classes
        # TODO: we're only looking at the training data so far
        self.available_classes = np.unique(self.y_train).tolist()

        # Split in test and train
        if self.x_test is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_train, self.y_train, test_size=self.test_split, random_state=self.random_state
            )

        # Split in train and validation
        if self.x_val is None:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_train, self.y_train, test_size=self.val_split, random_state=self.random_state
            )

        # Preprocess
        self._preprocess()

        # Note down the shape
        self.shape = self.x_train.shape[1:]

    ## Helpers
    def include_to_drop(self, include_data: Union[List[int], List[str]]) -> Union[List[int], List[str]]:
        """
        Convert a list of classes to include to a list of classes to drop
        :param include_data: classes to include
        :param all_classes: available classes
        :return: classes to drop
        """

        drop_classes = set(self.available_classes) - set(include_data)

        return list(drop_classes)

    def _get_data_set(self, data_split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the right data split
        :param data_split: train, val or test data?
        :return: the right data set
        """

        if data_split == "train":
            return self.x_train.copy(), self.y_train.copy()

        elif data_split == "test":
            return self.x_test.copy(), self.y_test.copy()

        elif data_split == "val":
            return self.x_val.copy(), self.y_val.copy()

        else:
            raise ValueError("The requested data must be of either train, val or test set.")

    @staticmethod
    def _ae_feature_selector(selected_layers: List[AutoencoderLayers], n_hidden: int) -> List[int]:
        """
        Index of features based on their name representation for symmetric autoencoders
        :param selected_layers: list of names for the desired layers
        :param n_hidden: number of hidden states
        :return: list of indices where to find the desired layers
        """
        # If nothing was specified, we'll assume that all features are meant
        if not selected_layers:
            return list(range(n_hidden))

        # If already numbers were given, use them
        if isinstance(selected_layers[0], int):
            return selected_layers

        # 0-indexed list are used
        n_hidden -= 1

        # We assume symmetric autoencoders, such that the code is in the middle
        i_code = math.floor(n_hidden / 2)

        # Life is easier with a translation dictionary
        trans_dict = {
            AutoencoderLayers.OUTPUT: [n_hidden],
            AutoencoderLayers.CODE: [i_code],
            AutoencoderLayers.ENCODER: list(range(i_code)),
            AutoencoderLayers.DECODER: list(range(i_code + 1, n_hidden)),
        }

        # We'll replace the selected lists by their index values a concatenate them
        index_list = [trans_dict[cur_el] for cur_el in selected_layers]
        index_list = [cur_el for cur_list in index_list for cur_el in cur_list]

        return sorted(index_list)

    def scikit_scale(self, scikit_scaler: Callable[[], TransformerMixin] = MinMaxScaler):
        """
        Apply a scikit scaler to the data, e.g. MinMaxScaler transform data to [0,1]
        :return:
        """
        # Fit scaler to train set
        self.scaler = scikit_scaler()
        self.x_train = self.scaler.fit_transform(self.x_train)

        # Scale the rest
        self.x_val = self.scaler.transform(self.x_val)
        self.x_test = self.scaler.transform(self.x_test)

        pass


class MNIST(DataLabels):
    def __init__(self, enrich_mnist_by=None, enrich_test_by=None, *args, **kwargs):
        """
        Load the MNIST data set
        """

        # Simply load the data with the kind help of Keras
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Add channel dimension to the data
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # If desired, add new samples from EMNIST to MNIST
        if enrich_mnist_by:

            # load both, train and test data set from EMNIST
            emnist_x_train, emnist_y_train = extract_samples('letters', 'train')
            emnist_x_test, emnist_y_test = extract_samples('letters', 'test')

            # Add channel dimension to emnist data
            emnist_x_train = np.expand_dims(emnist_x_train, -1)
            emnist_x_test = np.expand_dims(emnist_x_test, -1)

            # choose the desired letters from emnist and translate numerical lables to letters
            idx_train = []
            idx_test = []
            enrich_mnist_by = [i-9 for i in enrich_mnist_by]
            for i in range(len(enrich_mnist_by)):
                # get locations/indices of desired letters
                idx_train.append(np.where(emnist_y_train == list(enrich_mnist_by)[i]))
                idx_test.append(np.where(emnist_y_test == list(enrich_mnist_by)[i]))

            idx_train = np.asarray(idx_train).flatten()
            emnist_x_train = emnist_x_train[idx_train]
            emnist_y_train = emnist_y_train[idx_train]+9

            idx_test = np.asarray(idx_test).flatten()
            emnist_x_test = emnist_x_test[idx_test]
            emnist_y_test = emnist_y_test[idx_test]+9

            # concatenate mnist train set and emnist train dataset
            y_train = np.append(y_train, emnist_y_train)
            x_train = np.concatenate((x_train, emnist_x_train), axis=0)

            # concatenate mnist test set and emnist test dataset
            y_test = np.append(y_test, emnist_y_test)
            x_test = np.concatenate((x_test, emnist_x_test), axis=0)

        super(MNIST, self).__init__(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, *args, **kwargs
        )

    def _preprocess(self):
        """
        For MNIST, we can scale everything by just dividing by 255
        :return:
        """
        self.x_train = self.x_train / 255.
        self.x_test = self.x_test / 255.
        self.x_val = self.x_val / 255.


class EMNIST(DataLabels):
    def __init__(self, anom_list,  *args, **kwargs):
        """
        Load the MNIST data set
        """

        # load MNIST letters using emnist package
        data, labels = extract_samples('letters', 'train')

        # Add channel dimension to the data
        data = np.expand_dims(data, -1)

        # take anom_list as anomalies and delete other values and map to one value
        idx = np.where((labels >= anom_list[0]) & (labels <= anom_list[len(anom_list) - 1]))
        data = data[idx]
        labels = labels[idx]
        labels.fill(10)

        # load mnist digit dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Add channel dimension to the data
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # concatenate mnist and emnist dataset
        dat = np.concatenate((data, x_train, x_test), axis=0)
        label = np.concatenate((labels, y_train, y_test), axis=0)

        super(EMNIST, self).__init__(x_train=dat, y_train=label, *args,
                                     **kwargs)

    def _preprocess(self):
        """
        For MNIST, we can scale everything by just dividing by 255
        :return:
        """
        self.x_train = self.x_train / 255.
        self.x_test = self.x_test / 255.
        self.x_val = self.x_val / 255.


class CreditCard(DataLabels):
    def __init__(
            self, data_path: Path = (BASE_PATH / "data" / "creditcard" / "creditcard").with_suffix(".csv"),
            *args, **kwargs
    ):
        """
        Load the CreditCard data set (https://www.kaggle.com/mlg-ulb/creditcardfraud)
        :param data_path: absolute path to the CreditCard csv
        """

        data = pd.read_csv(data_path)

        # Time axis does not directly add information (although frequency might be a feature)
        data = data.drop(['Time'], axis=1)

        # Column class has the anomaly values, the rest is data
        x_train = data.drop(['Class'], axis=1)
        y_train = data.loc[:, ['Class']]

        # We don't need the overhead of pandas here
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()

        # TODO: why is this even in here?
        # for i in range(len(y_train)):
        #     y_train[i, 0] = y_train[i, 0].replace("\'", "")

        super(CreditCard, self).__init__(
            x_train=x_train, y_train=y_train, *args, **kwargs
        )

    def _preprocess(self):
        """
        Standardscale the data
        :return:
        """
        self.y_test = self.y_test.astype(np.int)
        self.y_train = self.y_train.astype(np.int)
        self.y_val = self.y_val.astype(np.int)
        self.scikit_scale()

    def _drop_class(self):
        """
        Drop frauds (Class==1)
        """

        # Delete from training data if we train the autoencoder
        if not self.is_alarm:
            self.x_train = np.delete(self.x_train, np.where(self.y_train == self.drop_num), axis=0)
            self.y_train = np.delete(self.y_train, np.where(self.y_train == self.drop_num), axis=0)

            # We should also drop it from the validation data, such that we only optimise on reconstruction valid data
            self.x_val = np.delete(self.x_val, np.where(self.y_val == self.drop_num), axis=0)
            self.y_val = np.delete(self.y_val, np.where(self.y_val == self.drop_num), axis=0)

        # Rewrite train labels -> not necessary for this data set
        if not self.is_binary:
            raise NotImplementedError("This data set only has binary labels")


class NSL_KDD(DataLabels):
    def __init__(self, data_folder: str = "NSL-KDD", *args, **kwargs):
        """
        NSL KDD data set: https://www.unb.ca/cic/datasets/nsl.html
        :param data_folder: subfolder of "data" where raw data resides
        """

        # Open raw data
        common_path = BASE_PATH / "data" / data_folder
        train_data = arff.loadarff((common_path / "KDDTrain+").with_suffix(".arff"))
        test_data = arff.loadarff((common_path / "KDDTest+").with_suffix(".arff"))

        # Extract column names
        all_cols = [cur_key for cur_key in test_data[1]._attributes.keys()]
        all_cat = {
            cur_key: cur_val.range for cur_key, cur_val in test_data[1]._attributes.items()
            if cur_val.range is not None
        }

        # Create pandas dataframe
        train_data = pd.DataFrame(data=train_data[0], columns=all_cols)
        test_data = pd.DataFrame(data=test_data[0], columns=all_cols)

        # Mark respective columns as categorical
        for cur_key, cur_val in all_cat.items():
            # We need to decode the byte strings first
            test_data[cur_key] = pd.Categorical(
                test_data[cur_key].str.decode('UTF-8'), categories=cur_val, ordered=False
            )
            train_data[cur_key] = pd.Categorical(
                train_data[cur_key].str.decode('UTF-8'), categories=cur_val, ordered=False
            )

        # For whatever reason, the anomaly labels are only in the .txt files... load them separately
        train_labels = pd.read_csv((common_path / "KDDTrain+").with_suffix(".txt"), header=None)
        train_labels = train_labels.iloc[:, -2].astype("category")
        train_labels = train_labels.map(self._attack_map())
        # NOTE: train_labels categories might not be mapped to the same number as in test_labels -> index by name
        test_labels = pd.read_csv((common_path / "KDDTest+").with_suffix(".txt"), header=None)
        test_labels = test_labels.iloc[:, -2].astype("category")
        test_labels = test_labels.map(self._attack_map())

        # Drop the class labels from the original data
        train_data = train_data.drop(columns="class")
        test_data = test_data.drop(columns="class")

        # Finally, 1-Hot encode the categorical data
        train_data = pd.get_dummies(train_data)
        test_data = pd.get_dummies(test_data)
        assert (train_data.columns == test_data.columns).all()

        # We'll use ndarrays from now on
        super(NSL_KDD, self).__init__(
            x_train=train_data.to_numpy(), y_train=train_labels.to_numpy(),
            x_test=test_data.to_numpy(), y_test=test_labels.to_numpy(), *args, **kwargs
        )

    def _attack_map(self) -> dict:
        """
        Map grouping the single attack classes
        :return: mapping dictionary
        """

        attack_dict = {
            'normal': 'normal',

            'back': 'DoS',
            'land': 'DoS',
            'neptune': 'DoS',
            'pod': 'DoS',
            'smurf': 'DoS',
            'teardrop': 'DoS',
            'mailbomb': 'DoS',
            'apache2': 'DoS',
            'processtable': 'DoS',
            'udpstorm': 'DoS',

            'ipsweep': 'Probe',
            'nmap': 'Probe',
            'portsweep': 'Probe',
            'satan': 'Probe',
            'mscan': 'Probe',
            'saint': 'Probe',

            'ftp_write': 'R2L',
            'guess_passwd': 'R2L',
            'imap': 'R2L',
            'multihop': 'R2L',
            'phf': 'R2L',
            'spy': 'R2L',
            'warezclient': 'R2L',
            'warezmaster': 'R2L',
            'sendmail': 'R2L',
            'named': 'R2L',
            'snmpgetattack': 'R2L',
            'snmpguess': 'R2L',
            'xlock': 'R2L',
            'xsnoop': 'R2L',
            'worm': 'R2L',

            'buffer_overflow': 'U2R',
            'loadmodule': 'U2R',
            'perl': 'U2R',
            'rootkit': 'U2R',
            'httptunnel': 'U2R',
            'ps': 'U2R',
            'sqlattack': 'U2R',
            'xterm': 'U2R'
        }

        return attack_dict

    def _preprocess(self):
        """
        Minmaxscale the data
        :return:
        """
        self.scikit_scale(scikit_scaler=MinMaxScaler)


class IDS(DataLabels):
    def __init__(
            self, start: int = None, stop: int = None,
            data_path: Path=(BASE_PATH / "data" / "IDS" / "ids").with_suffix(".h5"), *args, **kwargs
    ):
        """
        IDS data set: https://www.unb.ca/cic/datasets/ids-2018.html
        Download the data using awscli:
        aws s3 sync --no-sign-request --region eu-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" <dst-folder>
        :param start: first row number to read (for debugging)
        :param stop: last row number to read (for debugging)
        :param data_path: file with preprocessed data
        """

        # Open data
        if data_path.exists():
            df = pd.read_hdf(data_path, start=start, stop=stop)
        else:
            df = self.csv_to_h5(data_path=data_path)

        # Map the labels accordingly
        df["Label"] = df["Label"].map(self._attack_map())
        labels = df['Label']
        df = df.drop(columns='Label')

        # drop some object features if exist
        df = df.drop(columns=["Src IP", "Flow ID", "Dst IP"], errors="ignore")

        super(IDS, self).__init__(x_train=df, y_train=labels, *args, **kwargs)

    def _attack_map(self) -> dict:
        """
        Map grouping the single attack classes
        :return: mapping dictionary
        """

        attack_dict = {
            'Benign': 'Benign',

            'Bot': 'Bot',

            'FTP-BruteForce': 'BruteForce',
            'SSH-Bruteforce': 'BruteForce',

            'DDoS attacks-LOIC-HTTP': 'DDoS',
            'DDOS attack-LOIC-UDP': 'DDoS',
            'DDOS attack-HOIC': 'DDoS',

            'DoS attacks-GoldenEye': 'DoS',
            'DoS attacks-Slowloris': 'DoS',
            'DoS attacks-SlowHTTPTest': 'DoS',
            'DoS attacks-Hulk': 'DoS',

            'Infilteration': 'Infiltration',

            'Brute Force -Web': 'WebAttacks',
            'Brute Force -XSS': 'WebAttacks',
            'SQL Injection': 'WebAttacks',
        }

        return attack_dict

    def _type_map(self):

        return {
            "Dst Port": "integer",
            "Protocol": "integer",
            "Flow Duration": "integer",
            "Tot Fwd Pkts": "integer",
            "Tot Bwd Pkts": "integer",
            "TotLen Fwd Pkts": "integer",
            "TotLen Bwd Pkts": "integer",
            "Fwd Pkt Len Max": "integer",
            "Fwd Pkt Len Min": "integer",
            "Fwd Pkt Len Mean": "float",
            "Fwd Pkt Len Std": "float",
            "Bwd Pkt Len Max": "integer",
            "Bwd Pkt Len Min": "integer",
            "Bwd Pkt Len Mean": "float",
            "Bwd Pkt Len Std": "float",
            "Flow Byts/s": "float",
            "Flow Pkts/s": "float",
            "Flow IAT Mean": "float",
            "Flow IAT Std": "float",
            "Flow IAT Max": "integer",
            "Flow IAT Min": "integer",
            "Fwd IAT Tot": "integer",
            "Fwd IAT Mean": "float",
            "Fwd IAT Std": "float",
            "Fwd IAT Max": "integer",
            "Fwd IAT Min": "integer",
            "Bwd IAT Tot": "integer",
            "Bwd IAT Mean": "float",
            "Bwd IAT Std": "float",
            "Bwd IAT Max": "integer",
            "Bwd IAT Min": "integer",
            "Fwd PSH Flags": "integer",
            "Bwd PSH Flags": "integer",
            "Fwd URG Flags": "integer",
            "Bwd URG Flags": "integer",
            "Fwd Header Len": "integer",
            "Bwd Header Len": "integer",
            "Fwd Pkts/s": "float",
            "Bwd Pkts/s": "float",
            "Pkt Len Min": "integer",
            "Pkt Len Max": "integer",
            "Pkt Len Mean": "float",
            "Pkt Len Std": "float",
            "Pkt Len Var": "float",
            "FIN Flag Cnt": "integer",
            "SYN Flag Cnt": "integer",
            "RST Flag Cnt": "integer",
            "PSH Flag Cnt": "integer",
            "ACK Flag Cnt": "integer",
            "URG Flag Cnt": "integer",
            "CWE Flag Count": "integer",
            "ECE Flag Cnt": "integer",
            "Down/Up Ratio": "integer",
            "Pkt Size Avg": "float",
            "Fwd Seg Size Avg": "float",
            "Bwd Seg Size Avg": "float",
            "Fwd Byts/b Avg": "integer",
            "Fwd Pkts/b Avg": "integer",
            "Fwd Blk Rate Avg": "integer",
            "Bwd Byts/b Avg": "integer",
            "Bwd Pkts/b Avg": "integer",
            "Bwd Blk Rate Avg": "integer",
            "Subflow Fwd Pkts": "integer",
            "Subflow Fwd Byts": "integer",
            "Subflow Bwd Pkts": "integer",
            "Subflow Bwd Byts": "integer",
            "Init Fwd Win Byts": "integer",
            "Init Bwd Win Byts": "integer",
            "Fwd Act Data Pkts": "integer",
            "Fwd Seg Size Min": "integer",
            "Active Mean": "float",
            "Active Std": "float",
            "Active Max": "integer",
            "Active Min": "integer",
            "Idle Mean": "float",
            "Idle Std": "float",
            "Idle Max": "integer",
            "Idle Min": "integer"
        }

    def csv_to_h5(self, data_path: Path):
        """
        Open raw data, preprocess and save in single file.
        Note: all NaN & infinity rows are dropped.
        :param data_path: path to the data
        """
        print("We need to convert the raw data first. This might take some time.")

        # Look for all suitable raw files
        all_files = [cur_file for cur_file in data_path.parent.iterdir() if cur_file.suffix == ".csv"]

        # Combine to overall data
        all_df = pd.DataFrame()
        for cur_file in all_files:
            # Open the respective file
            cur_df = pd.read_csv(
                cur_file, header=0, parse_dates=["Timestamp"], index_col=["Timestamp"], low_memory=False, na_values="Infinity"
            )

            # For whatever reason, they repeat the header row within one csv file. Drop these.
            try:
                cur_df = cur_df.drop(index="Timestamp", errors="ignore")
            except TypeError:
                pass

            # Drop rows with NaN, infinity
            cur_df = cur_df.dropna()

            # Convert remaining types automatically; infer_object() only returns objects
            # TODO: would be even nicer to use unsigned ints
            type_map = self._type_map()
            for cur_col in cur_df.columns:
                if cur_col in type_map:
                    cur_df[cur_col] = pd.to_numeric(cur_df[cur_col], downcast=type_map[cur_col])

            all_df = pd.concat([all_df, cur_df], sort=False)

        # For whatever reason, there is not always a source port
        try:
            all_df["Src Port"] = pd.to_numeric(all_df["Src Port"], downcast="unsigned")
        except KeyError:
            pass
        # Category type also saves some space
        all_df["Protocol"] = all_df["Protocol"].astype("category")
        # One-hot encoding
        all_df = pd.get_dummies(all_df, columns=['Protocol'])

        # Save and return
        # TODO: get TypeError (expectred Col not ObjectAtom) when index is kept
        all_df.reset_index(drop=True).to_hdf(data_path, key="ids", format="table")
        return all_df

    def _preprocess(self):
        """
        Minmaxscale the data
        :return:
        """

        self.y_train = self.y_train.to_numpy()
        self.y_val = self.y_val.to_numpy()
        self.y_test = self.y_test.to_numpy()
        self.scikit_scale(scikit_scaler=MinMaxScaler)

    def _drop_class(self):
        """
        Drop intrusions
        :return:
        """
        switcher = {1: 'Bot', 2: 'BruteForce', 3: 'DDoS', 4: 'DoS', 5: 'Infilteration', 6: 'Web Attacks'}

        malicious_class = switcher.get(self.drop_num)
        self.y_train[self.y_train != malicious_class] = 0
        self.y_train[self.y_train == malicious_class] = 1
        self.y_train = self.y_train.astype(int)
        self.y_val[self.y_val != malicious_class] = 0
        self.y_val[self.y_val == malicious_class] = 1
        self.y_val = self.y_val.astype(int)
        # Delete from training data if we train the autoencoder
        if not self.is_alarm:
            self.x_train = np.delete(self.x_train, np.where(self.y_train == self.drop_num), axis=0)
        self.y_train = np.delete(self.y_train, np.where(self.y_train == self.drop_num), axis=0)

        # We should also drop it from the validation data, such that we only optimise on reconstruction valid data
        self.x_val = np.delete(self.x_val, np.where(self.y_val == self.drop_num), axis=0)
        self.y_val = np.delete(self.y_val, np.where(self.y_val == self.drop_num), axis=0)

        # Rewrite train labels -> not necessary for this data set
        if not self.is_binary:
            raise NotImplementedError("This data set only has binary labels")

