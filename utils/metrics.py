import numpy as np
import pandas as pd


def _confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def accuracy(y_true, y_pred):
    tp, fp, fn, tn = _confusion(y_true, y_pred)
    return (tp + tn) / len(y_true)


def precision(y_true, y_pred):
    tp, fp, _, _ = _confusion(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(y_true, y_pred):
    tp, _, fn, _ = _confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def auc(y_true, scores):
    ranks = pd.Series(scores).rank().to_numpy()
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
