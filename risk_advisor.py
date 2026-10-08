import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.ensemble import GradientBoostingClassifier


def features(logits):
    logits = np.asarray(logits, dtype=np.float64)
    p = softmax(logits, axis=1)
    ps = np.sort(p, axis=1)[:, ::-1]
    ls = np.sort(logits, axis=1)[:, ::-1]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    return np.column_stack([
        ps[:, :3],
        ps[:, 0] - ps[:, 1],
        entropy,
        ls[:, 0],
        ls[:, 0] - ls[:, 1],
        logits.std(axis=1),
        logsumexp(logits, axis=1),
    ])


class RiskAdvisor:
    def __init__(self, **kwargs):
        params = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                      subsample=0.8, random_state=0)
        params.update(kwargs)
        self.clf = GradientBoostingClassifier(**params)

    def fit(self, logits, correct):
        y = np.asarray(correct).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError('need both correct and incorrect samples to fit')
        self.clf.fit(features(logits), y)
        return self

    def score(self, logits):
        return self.clf.predict_proba(features(logits))[:, 1]
