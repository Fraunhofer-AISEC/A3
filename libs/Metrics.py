import numpy as np

from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from typing import Union, List

from libs.A3 import A3


def thresh_pred(in_pred: np.ndarray, thresh_pred: float = .5):
    """
    Make a binary predicition based on a threshold
    :param in_pred: original prediction with uncertainties
    :param thresh_pred: prediction threshold
    :return: binary prediction
    """
    out_pred = in_pred.copy()
    out_pred[in_pred >= thresh_pred] = 1
    out_pred[in_pred < thresh_pred] = 0

    return out_pred


def evaluate(a3: Union[A3, np.ndarray], test_alarm: tuple, threshold: float = 0.5):
    """
    Evaluate the performance of both the threshold and the alarm network based anomaly detection
    :param a3: network to be evaluated - or scores of already predicted data
    :param test_alarm: test data for network
    :param threshold: decision thresholds above which samples are considered anomalous (label 1)
    :return: list of results: f1, precision, recall
    """
    try:
        pred_y = a3.predict([test_alarm[0]], get_activation=True)
    except AttributeError:
        pred_y = a3

    pred_y = pred_y if not isinstance(pred_y, list) else pred_y[0]
    pred_y = thresh_pred(pred_y, threshold)

    # Our metrics
    pre_alarm = precision_score(test_alarm[1], pred_y)
    rec_alarm = recall_score(test_alarm[1], pred_y)
    f1_alarm = f1_score(test_alarm[1], pred_y)

    return [f1_alarm, pre_alarm, rec_alarm]


def evaluate_multiple(a3: Union[A3, np.ndarray], test_alarm: tuple, thresholds: List[float]):
    """
    Wrapper to evaluate the performance under multiple thresholds
    :param a3:
    :param test_alarm:
    :param thresholds:
    :return: [auc-roc, auc-pr, length(thresholds)*[f1, prec, recall]]
    """
    try:
        pred_y = a3.predict([test_alarm[0]], get_activation=True)
    except AttributeError:
        pred_y = a3

    # We first add some threshold-independent metrics...
    all_results = [
        roc_auc_score(y_true=test_alarm[1], y_score=pred_y),
        average_precision_score(y_true=test_alarm[1], y_score=pred_y),
    ]

    # ... and then for multiple thresholds
    for cur_tresh in thresholds:
        all_results.extend(
            evaluate(a3=pred_y, test_alarm=test_alarm, threshold=cur_tresh)
        )

    return all_results
