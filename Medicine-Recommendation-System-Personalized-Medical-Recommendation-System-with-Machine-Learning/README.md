# Medicine Recommendation System

This folder is the cleaned Python version of the original Kaggle notebook. It
uses symptom CSV data to predict a likely disease and then returns supporting
information such as description, precautions, medications, diet, workout, and
symptom severity.

## Files

- `recommendation_engine.py`: reusable prediction and recommendation engine.
- `predict_cli.py`: command-line disease predictor.
- `train_model.py`: model comparison script with classical ML plus ANN/MLP.
- `data_audit.py`: dataset quality, correlation, duplicate, and lookup-table audit.
- `Training.csv`: binary symptom matrix with `prognosis` labels.
- `description.csv`, `precautions_df.csv`, `medications.csv`, `diets.csv`,
  `workout_df.csv`, `Symptom-severity.csv`: recommendation lookup tables.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you already installed dependencies from `Medical Chatbot/requirements.txt`,
you do not need to install again.

## Predict From CLI

```bash
python predict_cli.py "itching, skin rash, nodal skin eruptions"
```

The engine accepts symptom names with spaces or underscores and uses fuzzy
matching for small typing differences. It also removes duplicate training rows
in memory and compares selected symptoms against cleaned disease symptom
templates, which gives clearer confidence for partial but coherent symptom
sets.

## Train And Evaluate

```bash
python train_model.py
```

This compares the notebook models and saves a trained Random Forest model at
`models/best_model.joblib`. The trainer canonicalizes disease names, removes
duplicate rows, and expands the data with safe partial-symptom examples derived
from existing disease templates. Ambiguous generated symptom patterns that map
to multiple diseases are removed.

The trainer evaluates:

- SVC
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Multinomial Naive Bayes
- ANN-style MLP classifier

On the current dataset, all models reach 100% cross-validation and holdout
accuracy. That does **not** mean the system is clinically production-grade; the
dataset has many duplicate/template rows, so random splits are very easy.

## Audit Dataset Quality

```bash
python data_audit.py
```

This creates `models/data_audit_report.md`. Key checks include:

- class balance
- exact duplicate rows
- duplicate symptom patterns
- highly correlated symptom features
- lookup-table coverage
- empty recommendation values

## Production-Grade Notes

The current engine is stronger than the original notebook because it now:

- loads a saved model from `models/best_model.joblib` when available
- falls back to training from CSV when no saved model exists
- cleans common disease spelling issues in output labels
- returns top predictions instead of only one disease
- returns confidence, reliability level, warnings, and safety notes
- flags emergency symptoms such as chest pain, breathlessness, coma, and blood
  in sputum

For real production medical diagnosis, you still need:

- independently collected real-world validation data
- clinician-reviewed medication and care recommendations
- calibration testing for predicted probabilities
- monitoring for unsafe outputs and uncertain cases
- clear “not medical advice” and emergency-care messaging

## Use In Python

```python
from recommendation_engine import MedicalRecommendationEngine

engine = MedicalRecommendationEngine()
result = engine.predict(["itching", "skin rash", "fatigue"])

print(result.disease)
print(result.description)
print(result.precautions)
```

## Medical Disclaimer

This system is for educational assistance only. It is not a medical device and
must not replace consultation with a qualified healthcare professional.
