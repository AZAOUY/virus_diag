import pandas as pd


class Preprocessor:
    """Imputation par la médiane + standardisation (appris sur le train uniquement)."""

    def __init__(self):
        self.columns = None
        self.medians = None
        self.means = None
        self.stds = None

    def fit(self, X: pd.DataFrame):
        self.columns = list(X.columns)
        self.medians = X.median()
        filled = X.fillna(self.medians)
        self.means = filled.mean()
        self.stds = filled.std().replace(0, 1)
        return self

    def transform(self, X: pd.DataFrame):
        X = X[self.columns].fillna(self.medians)
        return ((X - self.means) / self.stds).to_numpy(dtype=float)

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)
