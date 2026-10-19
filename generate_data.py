import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "age": rng.integers(1, 90, n),
    "temperature": rng.normal(37.5, 1.0, n).round(1),
    "cough": rng.integers(0, 2, n),
    "fatigue": rng.integers(0, 2, n),
    "oxygen_saturation": rng.normal(95, 3, n).round(1),
    "crp": np.abs(rng.normal(10, 8, n)).round(1),
})

z = (-1.0 + 1.5 * (df["temperature"] - 37.5) + 1.2 * df["cough"]
     + 0.8 * df["fatigue"] - 0.4 * (df["oxygen_saturation"] - 95)
     + 0.08 * (df["crp"] - 10))
proba = 1 / (1 + np.exp(-z))
df["infected"] = (rng.random(n) < proba).astype(int)

df.loc[rng.choice(n, 30, replace=False), "crp"] = np.nan   # valeurs manquantes
df = pd.concat([df, df.sample(10, random_state=1)])        # doublons

df.to_csv("data/patient_data.csv", index=False)
print("patient_data.csv généré :", df.shape)
