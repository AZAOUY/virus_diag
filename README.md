# virus_diag : diagnostic viral avec une architecture POO

Projet d'application du Chapitre 5 (POO adaptée à l'IA), Master IA, Génie Logiciel.

## Structure
```
virus_diag/
├── data/        patient_data.csv (données synthétiques)
├── core/        dataset, model (abstrait), logistic_regression, neural_network, optimizer
├── pipeline/    trainer, evaluator, predictor (ClinicalPredictor)
├── utils/       preprocessing, metrics
├── app/         api (Flask), interface_clinique
└── tests/       tests pytest
```

## Installation et lancement
```bash
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py           # crée data/patient_data.csv
python main.py                    # entraîne, évalue, sauvegarde le meilleur modèle
python -m app.interface_clinique  # interface console
python -m app.api                 # API : POST /diagnose
python -m pytest                  # tests
```

## Réponses à l'exercice
**1. ClinicalPredictor** : voir `pipeline/predictor.py`. Il reçoit un modèle déjà entraîné, expose `diagnose(patient_data)` et retourne "Infecté" si la prédiction est >= 0.5, "Sain" sinon.

**2. Concept POO** : le **polymorphisme**, rendu possible par l'**abstraction** (interface commune `predict()`). ClinicalPredictor appelle `model.predict()` sans connaître le type exact du modèle, donc les modèles sont interchangeables.

**3. Pourquoi séparer prédiction et entraînement ?** Principe de responsabilité unique (séparation des problèmes) :
- l'entraînement est lourd et ponctuel, la prédiction est légère et fréquente ;
- en production, le prédicteur n'a besoin que du modèle entraîné (pas des données d'apprentissage sensibles) ;
- on peut changer ou ré-entraîner le modèle sans toucher à la partie clinique ;
- on évite qu'une manipulation du prédicteur modifie les paramètres du modèle (encapsulation) ;
- chaque partie se teste séparément.
