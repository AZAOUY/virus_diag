import numpy as np
from core.model import Model


class GaussianNaiveBayes(Model):
    """Classifieur bayésien naïf gaussien (sain / infecté)."""

    def __init__(self, var_smoothing=1e-9):
        super().__init__()
        self.var_smoothing = var_smoothing
        self._params = None

    def train(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        eps = self.var_smoothing * X.var(axis=0).max()
        self._params = {}
        for c in (0, 1):
            Xc = X[y == c]
            self._params[c] = {
                "prior": len(Xc) / len(X),
                "mean": Xc.mean(axis=0),
                "var": Xc.var(axis=0) + eps,
            }
        self._trained = True
        return self

    def _log_likelihood(self, X, c):
        p = self._params[c]
        return (np.log(p["prior"])
                - 0.5 * np.sum(np.log(2 * np.pi * p["var"]))
                - 0.5 * np.sum((X - p["mean"]) ** 2 / p["var"], axis=1))

    def predict(self, X):
        """Retourne la probabilité d'être infecté pour chaque ligne de X."""
        if not self._trained:
            raise RuntimeError("Le modèle doit être entraîné avant predict().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._sigmoid(self._log_likelihood(X, 1) - self._log_likelihood(X, 0))