from __future__ import annotations

import ast
import difflib
from itertools import combinations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = DATA_DIR / "models" / "best_model.joblib"
EMERGENCY_SYMPTOMS = {
    "chest_pain",
    "breathlessness",
    "coma",
    "stomach_bleeding",
    "acute_liver_failure",
    "blood_in_sputum",
    "weakness_of_one_body_side",
    "altered_sensorium",
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


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _prettify_symptom(symptom: str) -> str:
    return symptom.replace("_", " ").strip().title()


def _parse_list_cell(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [part.strip() for part in text.split(",") if part.strip()]


def _canonical_disease_name(disease: str) -> str:
    return DISEASE_ALIASES.get(str(disease), str(disease).strip())


@dataclass(frozen=True)
class SymptomMatch:
    original: str
    matched: str | None
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    disease: str
    confidence: float | None
    top_predictions: list[tuple[str, float]]
    matched_symptoms: list[SymptomMatch]
    unknown_symptoms: list[str]
    description: str
    precautions: list[str]
    medications: list[str]
    diets: list[str]
    workouts: list[str]
    symptom_severity: dict[str, int]
    reliability: str
    warnings: list[str]
    safety_notes: list[str]


class MedicalRecommendationEngine:
    """CSV-backed symptom-to-disease recommendation engine."""

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        *,
        n_estimators: int = 160,
        random_state: int = 42,
        fuzzy_cutoff: float = 0.72,
        model_path: str | Path | None = DEFAULT_MODEL_PATH,
        min_symptoms_for_reliable_prediction: int = 3,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.fuzzy_cutoff = fuzzy_cutoff
        self.model_path = Path(model_path) if model_path else None
        self.min_symptoms_for_reliable_prediction = min_symptoms_for_reliable_prediction
        self._training_df: pd.DataFrame | None = None
        self._model = None
        self._symptom_columns: list[str] | None = None
        self._disease_templates: dict[str, list[set[str]]] | None = None

    @property
    def training_df(self) -> pd.DataFrame:
        if self._training_df is None:
            training_df = pd.read_csv(self.data_dir / "Training.csv")
            training_df["prognosis"] = training_df["prognosis"].map(_canonical_disease_name)
            self._training_df = training_df.drop_duplicates().reset_index(drop=True)
        return self._training_df

    @property
    def symptom_columns(self) -> list[str]:
        if self._symptom_columns is None:
            self._symptom_columns = [
                column for column in self.training_df.columns if column != "prognosis"
            ]
        return self._symptom_columns

    @property
    def model(self):
        if self._model is None:
            loaded_model = self._load_saved_model()
            if loaded_model is not None:
                self._model = loaded_model
                return self._model

            model_df = self._model_training_df()
            x_train = model_df[self.symptom_columns]
            y_train = model_df["prognosis"]
            self._model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            self._model.fit(x_train, y_train)
        return self._model

    def _model_training_df(self) -> pd.DataFrame:
        augmented_rows = [self.training_df]
        generated_rows: list[dict[str, object]] = []

        for _, row in self.training_df.iterrows():
            active_symptoms = [
                symptom
                for symptom in self.symptom_columns
                if int(row[symptom]) == 1
            ]
            disease = row["prognosis"]

            for size in range(3, min(6, len(active_symptoms)) + 1):
                for index, symptom_group in enumerate(combinations(active_symptoms, size)):
                    if index >= 16:
                        break
                    generated_rows.append(self._symptom_row(symptom_group, disease))

            if len(active_symptoms) > 6:
                for symptom_group in combinations(active_symptoms, len(active_symptoms) - 1):
                    generated_rows.append(self._symptom_row(symptom_group, disease))
                    if len(generated_rows) >= 8000:
                        break

        if generated_rows:
            augmented_rows.append(pd.DataFrame(generated_rows))

        model_df = pd.concat(augmented_rows, ignore_index=True).drop_duplicates()
        unique_labels_per_pattern = model_df.groupby(self.symptom_columns)["prognosis"].transform("nunique")
        return model_df[unique_labels_per_pattern == 1].drop_duplicates().reset_index(drop=True)

    def _symptom_row(self, symptoms: Iterable[str], disease: str) -> dict[str, object]:
        row: dict[str, object] = {symptom: 0 for symptom in self.symptom_columns}
        for symptom in symptoms:
            row[symptom] = 1
        row["prognosis"] = disease
        return row

    def _load_saved_model(self):
        if not self.model_path or not self.model_path.exists():
            return None

        package = joblib.load(self.model_path)
        if not isinstance(package, dict) or "model" not in package:
            return None

        saved_symptoms = package.get("symptoms")
        if saved_symptoms and list(saved_symptoms) != self.symptom_columns:
            return None

        return package["model"]

    def symptom_options(self) -> list[dict[str, str]]:
        return [
            {"value": symptom, "label": _prettify_symptom(symptom)}
            for symptom in self.symptom_columns
        ]

    def disease_options(self) -> list[str]:
        return sorted(self.training_df["prognosis"].unique().tolist())

    @property
    def disease_templates(self) -> dict[str, list[set[str]]]:
        if self._disease_templates is None:
            templates: dict[str, list[set[str]]] = {}
            for disease, group in self.training_df.groupby("prognosis"):
                unique_patterns = group[self.symptom_columns].drop_duplicates()
                templates[disease] = [
                    {
                        symptom
                        for symptom in self.symptom_columns
                        if int(row[symptom]) == 1
                    }
                    for _, row in unique_patterns.iterrows()
                ]
            self._disease_templates = templates
        return self._disease_templates

    def match_symptoms(self, symptoms: Iterable[str]) -> list[SymptomMatch]:
        symptom_lookup = {_normalise_key(symptom): symptom for symptom in self.symptom_columns}
        known_keys = list(symptom_lookup.keys())
        matches: list[SymptomMatch] = []

        for raw_symptom in symptoms:
            original = str(raw_symptom).strip()
            if not original:
                continue

            key = _normalise_key(original)
            if key in symptom_lookup:
                matches.append(
                    SymptomMatch(original=original, matched=symptom_lookup[key], confidence=1.0)
                )
                continue

            close = difflib.get_close_matches(key, known_keys, n=1, cutoff=self.fuzzy_cutoff)
            if close:
                ratio = difflib.SequenceMatcher(None, key, close[0]).ratio()
                matches.append(
                    SymptomMatch(
                        original=original,
                        matched=symptom_lookup[close[0]],
                        confidence=ratio,
                    )
                )
            else:
                matches.append(SymptomMatch(original=original, matched=None, confidence=0.0))

        return matches

    def predict(self, symptoms: Iterable[str], *, top_n: int = 5) -> PredictionResult:
        matched_symptoms = self.match_symptoms(symptoms)
        known_symptoms = sorted(
            {match.matched for match in matched_symptoms if match.matched is not None}
        )
        unknown_symptoms = [
            match.original for match in matched_symptoms if match.matched is None
        ]

        if not known_symptoms:
            raise ValueError("No recognised symptoms were provided.")

        input_row = pd.DataFrame([[0] * len(self.symptom_columns)], columns=self.symptom_columns)
        for symptom in known_symptoms:
            input_row.at[0, symptom] = 1

        model_predictions = self._model_predictions(input_row, top_n)
        template_predictions = self._template_predictions(known_symptoms, top_n)

        if template_predictions:
            disease, confidence = template_predictions[0]
            top_predictions = template_predictions
        else:
            disease, confidence = model_predictions[0]
            top_predictions = model_predictions

        recommendations = self.recommendations_for(disease)
        warnings = self._prediction_warnings(
            known_symptoms=known_symptoms,
            unknown_symptoms=unknown_symptoms,
            confidence=confidence,
            top_predictions=top_predictions,
        )
        safety_notes = self._safety_notes(known_symptoms)
        return PredictionResult(
            disease=disease,
            confidence=confidence,
            top_predictions=top_predictions,
            matched_symptoms=matched_symptoms,
            unknown_symptoms=unknown_symptoms,
            symptom_severity=self.symptom_severity(known_symptoms),
            reliability=self._reliability_label(warnings),
            warnings=warnings,
            safety_notes=safety_notes,
            **recommendations,
        )

    def _model_predictions(
        self,
        input_row: pd.DataFrame,
        top_n: int,
    ) -> list[tuple[str, float]]:
        if not hasattr(self.model, "predict_proba"):
            disease = _canonical_disease_name(str(self.model.predict(input_row)[0]))
            return [(disease, 1.0)]

        probabilities = self.model.predict_proba(input_row)[0]
        return sorted(
            [
                (_canonical_disease_name(str(label)), float(probability))
                for label, probability in zip(self.model.classes_, probabilities)
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]

    def _template_predictions(
        self,
        known_symptoms: list[str],
        top_n: int,
    ) -> list[tuple[str, float]]:
        selected = set(known_symptoms)
        if not selected:
            return []

        scored_diseases: list[tuple[str, float]] = []
        for disease, templates in self.disease_templates.items():
            best_score = 0.0
            for template in templates:
                if not template:
                    continue

                overlap = len(selected.intersection(template))
                if overlap == 0:
                    continue

                precision = overlap / len(selected)
                recall = overlap / len(template)
                jaccard = overlap / len(selected.union(template))
                score = (0.68 * precision) + (0.27 * recall) + (0.05 * jaccard)
                best_score = max(best_score, score)

            if best_score > 0:
                scored_diseases.append((disease, round(min(best_score, 1.0), 4)))

        return sorted(scored_diseases, key=lambda item: item[1], reverse=True)[:top_n]

    def _prediction_warnings(
        self,
        *,
        known_symptoms: list[str],
        unknown_symptoms: list[str],
        confidence: float | None,
        top_predictions: list[tuple[str, float]],
    ) -> list[str]:
        warnings: list[str] = []

        if len(known_symptoms) < self.min_symptoms_for_reliable_prediction:
            warnings.append(
                f"Only {len(known_symptoms)} recognised symptom(s) were provided; "
                "add more symptoms for a more reliable result."
            )

        if unknown_symptoms:
            warnings.append(
                "Some symptoms were not recognised and were ignored: "
                + ", ".join(unknown_symptoms)
            )

        if confidence is not None and confidence < 0.75:
            warnings.append(
                "The top prediction has lower confidence; review the probability and alternate predictions."
            )

        if len(top_predictions) >= 2:
            gap = top_predictions[0][1] - top_predictions[1][1]
            if gap < 0.15:
                warnings.append(
                    "The top predictions are close together, so this case is ambiguous."
                )

        return warnings

    def _reliability_label(self, warnings: list[str]) -> str:
        if not warnings:
            return "high"
        if len(warnings) <= 2:
            return "moderate"
        return "low"

    def _safety_notes(self, known_symptoms: list[str]) -> list[str]:
        if EMERGENCY_SYMPTOMS.intersection(known_symptoms):
            return [
                "Potential emergency symptoms were selected. Seek urgent medical care if symptoms are severe, sudden, or worsening."
            ]
        return [
            "This result is educational support only and should not replace professional diagnosis."
        ]

    def recommendations_for(self, disease: str) -> dict[str, object]:
        raw_or_canonical = self._raw_lookup_disease(disease)
        return {
            "description": self._single_lookup("description.csv", raw_or_canonical, "Description"),
            "precautions": self._precautions(raw_or_canonical),
            "medications": self._list_lookup("medications.csv", raw_or_canonical, "Medication"),
            "diets": self._list_lookup("diets.csv", raw_or_canonical, "Diet"),
            "workouts": self._workouts(raw_or_canonical),
        }

    def _raw_lookup_disease(self, disease: str) -> str:
        for raw_name, canonical_name in DISEASE_ALIASES.items():
            if _normalise_key(canonical_name) == _normalise_key(disease):
                return raw_name
        return disease

    def symptom_severity(self, symptoms: Iterable[str]) -> dict[str, int]:
        severity_df = pd.read_csv(self.data_dir / "Symptom-severity.csv")
        severity_df["_key"] = severity_df["Symptom"].astype(str).map(_normalise_key)
        result: dict[str, int] = {}

        for symptom in symptoms:
            match = severity_df[severity_df["_key"] == _normalise_key(symptom)]
            if not match.empty:
                result[symptom] = int(match.iloc[0]["weight"])

        return result

    def _read_csv(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / filename)

    def _find_column(self, df: pd.DataFrame, wanted: str) -> str | None:
        wanted_key = _normalise_key(wanted)
        for column in df.columns:
            if _normalise_key(column) == wanted_key:
                return column
        return None

    def _find_disease_row(self, df: pd.DataFrame, disease: str):
        disease_column = self._find_column(df, "Disease")
        if disease_column is None:
            return None

        disease_key = _normalise_key(disease)
        keys = df[disease_column].astype(str).map(_normalise_key)
        exact = df[keys == disease_key]
        if not exact.empty:
            return exact.iloc[0]

        close = difflib.get_close_matches(disease_key, keys.tolist(), n=1, cutoff=0.84)
        if close:
            return df[keys == close[0]].iloc[0]

        return None

    def _single_lookup(self, filename: str, disease: str, value_column: str) -> str:
        df = self._read_csv(filename)
        row = self._find_disease_row(df, disease)
        column = self._find_column(df, value_column)
        if row is None or column is None or pd.isna(row[column]):
            return ""
        return str(row[column]).strip()

    def _list_lookup(self, filename: str, disease: str, value_column: str) -> list[str]:
        return _parse_list_cell(self._single_lookup(filename, disease, value_column))

    def _precautions(self, disease: str) -> list[str]:
        df = self._read_csv("precautions_df.csv")
        row = self._find_disease_row(df, disease)
        if row is None:
            return []

        precautions: list[str] = []
        for column in df.columns:
            if column.lower().startswith("precaution") and pd.notna(row[column]):
                value = str(row[column]).strip()
                if value:
                    precautions.append(value)
        return precautions

    def _workouts(self, disease: str) -> list[str]:
        df = self._read_csv("workout_df.csv")
        disease_column = self._find_column(df, "disease")
        workout_column = self._find_column(df, "workout")
        if disease_column is None or workout_column is None:
            return []

        disease_key = _normalise_key(disease)
        keys = df[disease_column].astype(str).map(_normalise_key)
        matches = df[keys == disease_key]
        if matches.empty:
            close = difflib.get_close_matches(disease_key, keys.unique().tolist(), n=1, cutoff=0.84)
            if close:
                matches = df[keys == close[0]]

        return [
            str(value).strip()
            for value in matches[workout_column].dropna().tolist()
            if str(value).strip()
        ]
