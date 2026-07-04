from pathlib import Path
from itertools import combinations
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


DATA_DIR = Path(__file__).resolve().parent
MODEL_DIR = DATA_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
MODEL_TIEBREAK_PRIORITY = {
    "RandomForest": 5,
    "SVC": 4,
    "MultinomialNB": 3,
    "ANN_MLP": 2,
    "KNeighbors": 1,
    "GradientBoosting": 0,
}
DISEASE_ALIASES = {
    "Diabetes ": "Diabetes",
    "Hypertension ": "Hypertension",
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids (piles)",
    "Osteoarthristis": "Osteoarthritis",
    "(vertigo) Paroymsal  Positional Vertigo": "(vertigo) Paroxysmal Positional Vertigo",
    "(vertigo) Paroymsal Positional Vertigo": "(vertigo) Paroxysmal Positional Vertigo",
}

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.",
)


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(DATA_DIR / "Training.csv")
    dataset["prognosis"] = dataset["prognosis"].map(
        lambda value: DISEASE_ALIASES.get(str(value), str(value).strip())
    )
    dataset = dataset.drop_duplicates().reset_index(drop=True)
    dataset = expand_partial_symptom_patterns(dataset)
    return dataset.drop(columns=["prognosis"]), dataset["prognosis"]


def expand_partial_symptom_patterns(dataset: pd.DataFrame) -> pd.DataFrame:
    symptom_columns = [column for column in dataset.columns if column != "prognosis"]
    generated_rows: list[dict[str, object]] = []

    for _, row in dataset.iterrows():
        active_symptoms = [
            symptom
            for symptom in symptom_columns
            if int(row[symptom]) == 1
        ]
        disease = row["prognosis"]

        for size in range(3, min(6, len(active_symptoms)) + 1):
            for index, symptom_group in enumerate(combinations(active_symptoms, size)):
                if index >= 16:
                    break
                generated_rows.append(symptom_row(symptom_columns, symptom_group, disease))

    if generated_rows:
        dataset = pd.concat([dataset, pd.DataFrame(generated_rows)], ignore_index=True)

    dataset = dataset.drop_duplicates()
    unique_labels_per_pattern = dataset.groupby(symptom_columns)["prognosis"].transform("nunique")
    return dataset[unique_labels_per_pattern == 1].drop_duplicates().reset_index(drop=True)


def symptom_row(
    symptom_columns: list[str],
    symptoms: tuple[str, ...],
    disease: str,
) -> dict[str, object]:
    row: dict[str, object] = {symptom: 0 for symptom in symptom_columns}
    for symptom in symptoms:
        row[symptom] = 1
    row["prognosis"] = disease
    return row


def build_models() -> dict[str, object]:
    return {
        "SVC": SVC(kernel="linear", probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=240,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5),
        "MultinomialNB": MultinomialNB(),
        "ANN_MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=350,
            early_stopping=False,
            random_state=42,
        ),
    }


def evaluate_models() -> tuple[str, object, pd.DataFrame, pd.Series, dict[str, dict[str, float]]]:
    x, y = load_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    models = build_models()
    cross_validator = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    metrics: dict[str, dict[str, float]] = {}

    print("Model Accuracy")
    print("==============")
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        holdout_accuracy = accuracy_score(y_test, predictions)
        cv_scores = cross_val_score(
            model,
            x,
            y,
            cv=cross_validator,
            scoring="accuracy",
            n_jobs=-1,
        )
        metrics[name] = {
            "holdout_accuracy": float(holdout_accuracy),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
        }
        print(
            f"{name}: holdout={holdout_accuracy:.4f}, "
            f"cv_mean={cv_scores.mean():.4f}, cv_std={cv_scores.std():.4f}"
        )

    best_model_name = max(
        metrics,
        key=lambda key: (
            metrics[key]["cv_accuracy_mean"],
            metrics[key]["holdout_accuracy"],
            MODEL_TIEBREAK_PRIORITY.get(key, 0),
        ),
    )
    return best_model_name, models[best_model_name], x, y, metrics


def train_and_save_best_model() -> None:
    best_model_name, model, x, y, metrics = evaluate_models()
    model.fit(x, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": best_model_name,
            "symptoms": x.columns.tolist(),
            "diseases": sorted(y.unique().tolist()),
            "metrics": metrics[best_model_name],
        },
        BEST_MODEL_PATH,
    )
    pd.Series({"best_model": best_model_name}).to_json(METRICS_PATH)
    pd.DataFrame(metrics).T.to_json(METRICS_PATH, orient="index", indent=2)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {BEST_MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")

    predictions = model.predict(x)
    print("\nFull-data Classification Report")
    print("===============================")
    print(classification_report(y, predictions))


def main() -> None:
    train_and_save_best_model()


if __name__ == "__main__":
    main()
