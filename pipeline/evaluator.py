from utils import metrics


class Evaluator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate(self, model, X, y):
        proba = model.predict(X)
        pred = (proba >= self.threshold).astype(int)
        return {
            "accuracy": metrics.accuracy(y, pred),
            "precision": metrics.precision(y, pred),
            "recall": metrics.recall(y, pred),
            "f1": metrics.f1_score(y, pred),
            "auc": metrics.auc(y, proba),
        }
