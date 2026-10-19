from abc import ABC, abstractmethod
import pickle
import numpy as np


class Model(ABC):
    """Interface commune à tous les modèles IA (abstraction)."""

    def __init__(self):
        self._trained = False

    @abstractmethod
    def train(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        """Retourne la probabilité d'être infecté pour chaque ligne de X."""
        ...

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _loss(y, p):
        eps = 1e-9
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
