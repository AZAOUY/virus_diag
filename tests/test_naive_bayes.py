import numpy as np
from core.naive_bayes import GaussianNaiveBayes


def test_naive_bayes_apprend():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(float)
    model = GaussianNaiveBayes().train(X, y)
    assert ((model.predict(X) >= 0.5) == y).mean() > 0.9