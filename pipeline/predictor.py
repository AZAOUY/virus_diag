import numpy as np
import pandas as pd


class ClinicalPredictor:
    """Diagnostique un patient avec n'importe quel modèle entraîné ayant predict()."""

    def __init__(self, model, preprocessor=None, threshold=0.5):
        self.model = model                  # modèle DÉJÀ entraîné
        self.preprocessor = preprocessor    # optionnel
        self.threshold = threshold

    def diagnose(self, patient_data):
        X = patient_data
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        prediction = float(self.model.predict(X)[0])
        return "Infecté" if prediction >= self.threshold else "Sain"
