import numpy as np
import sklearn.metrics as skm


def get_auroc(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_det_accuracy(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    fpr, tpr, thresholds = skm.roc_curve(labels, data)
    return .5 * (tpr + 1. - fpr).max()


def get_aupr_out(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr_out = skm.average_precision_score(labels, data)
    return aupr_out


def get_aupr_in(xin, xood):
    labels = [1] * len(xin) + [0] * len(xood)
    data = np.concatenate((-xin, -xood))
    aupr_in = skm.average_precision_score(labels, data)
    return aupr_in


def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)
