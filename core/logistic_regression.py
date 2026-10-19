import numpy as np
from core.model import Model
from core.optimizer import GradientDescent


class LogisticRegression(Model):
    def __init__(self, optimizer=None, epochs=500):
        super().__init__()
        self.optimizer = optimizer or GradientDescent(lr=0.1)
        self.epochs = epochs
        self._params = None
        self.losses = []

    def train(self, X, y):
        n, d = X.shape
        self._params = {"w": np.zeros(d), "b": np.zeros(1)}
        self.losses = []
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self._params["w"] + self._params["b"])
            grads = {"w": X.T @ (p - y) / n, "b": np.array([np.mean(p - y)])}
            self.optimizer.update(self._params, grads)
            self.losses.append(self._loss(y, p))
        self._trained = True
        return self

    def predict(self, X):
        return self._sigmoid(X @ self._params["w"] + self._params["b"])
