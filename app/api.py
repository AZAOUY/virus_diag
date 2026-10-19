import pickle
from flask import Flask, request, jsonify
from pipeline.predictor import ClinicalPredictor


def create_app(model_path="models/best_model.pkl"):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    predictor = ClinicalPredictor(bundle["model"], bundle["preprocessor"])

    app = Flask(__name__)

    @app.post("/diagnose")
    def diagnose():
        try:
            return jsonify({"diagnostic": predictor.diagnose(request.get_json())})
        except (KeyError, TypeError, ValueError):
            return jsonify({"erreur": "Données patient invalides ou incomplètes"}), 400

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
