class Trainer:
    """Responsable UNIQUEMENT de l'entraînement."""

    def __init__(self, model):
        self.model = model

    def run(self, X_train, y_train):
        self.model.train(X_train, y_train)
        return self.model
