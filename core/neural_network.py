import numpy as np
from core.model import Model
from core.optimizer import GradientDescent


class NeuralNetwork(Model):
    """Réseau de neurones à une couche cachée."""

    def __init__(self, hidden=8, optimizer=None, epochs=1000, seed=42):
        super().__init__()
        self.hidden = hidden
        self.optimizer = optimizer or GradientDescent(lr=0.1)
        self.epochs = epochs
        self.seed = seed
        self._params = None
        self.losses = []

    def _forward(self, X):
        p = self._params
        A1 = np.tanh(X @ p["W1"] + p["b1"])
        A2 = self._sigmoid(A1 @ p["W2"] + p["b2"])
        return A1, A2

    def train(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        y = y.reshape(-1, 1)
        self._params = {
            "W1": rng.normal(0, 0.5, (d, self.hidden)), "b1": np.zeros(self.hidden),
            "W2": rng.normal(0, 0.5, (self.hidden, 1)), "b2": np.zeros(1),
        }
        self.losses = []
        for _ in range(self.epochs):
            A1, A2 = self._forward(X)
            dZ2 = (A2 - y) / n
            dZ1 = (dZ2 @ self._params["W2"].T) * (1 - A1 ** 2)
            grads = {"W2": A1.T @ dZ2, "b2": dZ2.sum(axis=0),
                     "W1": X.T @ dZ1, "b1": dZ1.sum(axis=0)}
            self.optimizer.update(self._params, grads)
            self.losses.append(self._loss(y, A2))
        self._trained = True
        return self

    def predict(self, X):
        return self._forward(X)[1].ravel()
