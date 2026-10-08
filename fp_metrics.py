import numpy as np
from sklearn import metrics


def msp(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).max(axis=1)


def rc_curve(score, correct):
    order = np.argsort(-np.asarray(score, dtype=np.float64))
    c = np.asarray(correct)[order]
    n = len(c)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(1 - c) / np.arange(1, n + 1)
    return coverage, risk


def aurc_eaurc(score, correct):
    _, risk = rc_curve(score, correct)
    aurc = float(risk.mean())
    r = float(risk[-1])
    if r >= 1.0:
        return aurc, float('nan')
    optimal = r + (1 - r) * np.log(1 - r)
    return aurc, aurc - optimal


def fpr_at_95tpr(score, correct):
    fpr, tpr, _ = metrics.roc_curve(correct, score)
    return float(fpr[np.argmax(tpr >= 0.95)])


def evaluate(score, correct):
    score = np.asarray(score, dtype=np.float64)
    correct = np.asarray(correct).astype(int)
    aurc, eaurc = aurc_eaurc(score, correct)
    return {
        'acc': 100 * float(correct.mean()),
        'auroc': 100 * float(metrics.roc_auc_score(correct, score)),
        'aupr_err': 100 * float(metrics.average_precision_score(1 - correct, -score)),
        'aurc': 1000 * aurc,
        'eaurc': 1000 * eaurc,
        'fpr95': 100 * fpr_at_95tpr(score, correct),
    }
