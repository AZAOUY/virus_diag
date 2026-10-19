import numpy as np
from pipeline.predictor import ClinicalPredictor
from core.logistic_regression import LogisticRegression


class FakeModel:                         # aucun lien avec Model !
    def __init__(self, p):
        self.p = p

    def predict(self, X):
        return np.array([self.p])


def test_infecte():
    assert ClinicalPredictor(FakeModel(0.8)).diagnose([[1, 2]]) == "Infecté"


def test_sain():
    assert ClinicalPredictor(FakeModel(0.2)).diagnose([[1, 2]]) == "Sain"


def test_seuil_exactement_0_5():
    assert ClinicalPredictor(FakeModel(0.5)).diagnose([[1, 2]]) == "Infecté"


def test_regression_logistique_apprend():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(float)
    model = LogisticRegression(epochs=300).train(X, y)
    assert ((model.predict(X) >= 0.5) == y).mean() > 0.9
