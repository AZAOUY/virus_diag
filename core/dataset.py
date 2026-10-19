import pandas as pd


class Dataset:
    def __init__(self, path, target="infected"):
        self._path = path
        self._target = target
        self._df = None

    def load(self):
        df = pd.read_csv(self._path)
        df = df.dropna(subset=[self._target]).drop_duplicates()
        self._df = df.reset_index(drop=True)
        return self

    def split(self, test_size=0.2, seed=42):
        df = self._df.sample(frac=1, random_state=seed)
        n_test = int(len(df) * test_size)
        test, train = df.iloc[:n_test], df.iloc[n_test:]
        return (train.drop(columns=self._target), test.drop(columns=self._target),
                train[self._target].to_numpy(float), test[self._target].to_numpy(float))
