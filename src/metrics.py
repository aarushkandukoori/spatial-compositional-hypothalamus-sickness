'''Evaluation metrics for SYNTHETIC-data experiments (PROPOSAL reference code).'''
from __future__ import annotations

import numpy as np


def pr_auc(truth: np.ndarray, score: np.ndarray) -> float:
    '''Average precision of `score` for detecting truth==True (NaN if no positives).'''
    truth = np.asarray(truth, dtype=bool).ravel()
    score = np.asarray(score, dtype=float).ravel()
    if truth.sum() == 0:
        return float('nan')
    order = np.argsort(-score, kind='stable')
    t = truth[order]
    tp = np.cumsum(t)
    prec = tp / np.arange(1, len(t) + 1)
    return float(prec[t].mean())


def precision_recall(truth: np.ndarray, pred: np.ndarray):
    '''Precision and recall of a boolean prediction mask.'''
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = float((truth & pred).sum())
    prec = tp / pred.sum() if pred.sum() else float('nan')
    rec = tp / truth.sum() if truth.sum() else float('nan')
    return prec, rec


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def false_positive_rate(truth: np.ndarray, pred: np.ndarray) -> float:
    '''Fraction of true-null cells that are flagged (use on null synthetic data).'''
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    nulls = ~truth
    return float((pred & nulls).sum() / nulls.sum()) if nulls.sum() else float('nan')
