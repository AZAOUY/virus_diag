import os
import pickle

from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from utils.preprocessing import Preprocessor

dataset = Dataset("data/patient_data.csv", target="infected").load()
X_train, X_test, y_train, y_test = dataset.split()

prep = Preprocessor().fit(X_train)            # appris sur le train uniquement
Xtr, Xte = prep.transform(X_train), prep.transform(X_test)

models = {"logistic_regression": LogisticRegression(),
          "neural_network": NeuralNetwork()}
results = {}
for name, model in models.items():
    Trainer(model).run(Xtr, y_train)
    results[name] = Evaluator().evaluate(model, Xte, y_test)
    print(name, {k: round(v, 3) for k, v in results[name].items()})

best = max(results, key=lambda n: results[n]["f1"])
print("Meilleur modèle :", best)

os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump({"model": models[best], "preprocessor": prep}, f)
