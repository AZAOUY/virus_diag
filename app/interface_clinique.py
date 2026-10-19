import pickle
from pipeline.predictor import ClinicalPredictor


def main():
    with open("models/best_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    predictor = ClinicalPredictor(bundle["model"], bundle["preprocessor"])

    print("=== Interface clinique : diagnostic viral ===")
    patient = {col: float(input(f"{col} : ")) for col in bundle["preprocessor"].columns}
    print("Diagnostic :", predictor.diagnose(patient))


if __name__ == "__main__":
    main()
