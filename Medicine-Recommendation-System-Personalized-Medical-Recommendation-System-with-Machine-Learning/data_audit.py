from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
REPORT_PATH = DATA_DIR / "models" / "data_audit_report.md"
DISEASE_ALIASES = {
    "Diabetes ": "Diabetes",
    "Hypertension ": "Hypertension",
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids (piles)",
    "Osteoarthristis": "Osteoarthritis",
    "(vertigo) Paroymsal  Positional Vertigo": "(vertigo) Paroxysmal Positional Vertigo",
    "(vertigo) Paroymsal Positional Vertigo": "(vertigo) Paroxysmal Positional Vertigo",
}


def normalise(value: str) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def canonical_disease(value: str) -> str:
    return DISEASE_ALIASES.get(str(value), str(value).strip())


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Training.csv")


def high_correlation_pairs(features: pd.DataFrame, threshold: float = 0.92) -> list[tuple[str, str, float]]:
    corr = features.corr().abs()
    pairs: list[tuple[str, str, float]] = []
    columns = corr.columns.tolist()
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = corr.at[left, right]
            if pd.notna(value) and value >= threshold:
                pairs.append((left, right, float(value)))
    return sorted(pairs, key=lambda item: item[2], reverse=True)


def lookup_coverage(diseases: list[str]) -> list[str]:
    lookup_files = {
        "description.csv": "Disease",
        "precautions_df.csv": "Disease",
        "medications.csv": "Disease",
        "diets.csv": "Disease",
        "workout_df.csv": "disease",
    }

    disease_keys = {normalise(canonical_disease(disease)) for disease in diseases}
    lines: list[str] = []
    for filename, disease_column in lookup_files.items():
        df = pd.read_csv(DATA_DIR / filename)
        available = {
            normalise(canonical_disease(value))
            for value in df[disease_column].dropna().astype(str)
        }
        missing = sorted(
            disease
            for disease in diseases
            if normalise(canonical_disease(disease)) not in available
        )
        lines.append(
            f"- `{filename}`: {len(disease_keys) - len(missing)}/{len(disease_keys)} diseases covered"
        )
        if missing:
            lines.append(f"  Missing: {', '.join(missing)}")
    return lines


def empty_recommendation_cells() -> list[str]:
    checks = {
        "description.csv": ["Description"],
        "precautions_df.csv": ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"],
        "medications.csv": ["Medication"],
        "diets.csv": ["Diet"],
        "workout_df.csv": ["workout"],
    }

    lines: list[str] = []
    for filename, columns in checks.items():
        df = pd.read_csv(DATA_DIR / filename)
        for column in columns:
            if column not in df.columns:
                lines.append(f"- `{filename}` missing expected column `{column}`")
                continue
            missing_count = int(df[column].isna().sum() + (df[column].astype(str).str.strip() == "").sum())
            if missing_count:
                lines.append(f"- `{filename}.{column}` has {missing_count} empty value(s)")
    return lines or ["- No empty recommendation cells found in checked columns."]


def build_report() -> str:
    dataset = load_training_data()
    features = dataset.drop(columns=["prognosis"])
    labels = dataset["prognosis"]
    class_counts = labels.value_counts().sort_index()

    duplicate_rows = int(dataset.duplicated().sum())
    duplicate_feature_patterns = int(features.duplicated().sum())
    conflicting_patterns = int(
        dataset.groupby(features.columns.tolist())["prognosis"].nunique().gt(1).sum()
    )
    correlation_pairs = high_correlation_pairs(features)

    lines = [
        "# Data Audit Report",
        "",
        "## Dataset Shape",
        f"- Rows: {len(dataset)}",
        f"- Symptoms/features: {features.shape[1]}",
        f"- Disease classes: {labels.nunique()}",
        "",
        "## Class Balance",
        f"- Minimum rows per disease: {int(class_counts.min())}",
        f"- Maximum rows per disease: {int(class_counts.max())}",
        f"- Balanced classes: {'yes' if class_counts.min() == class_counts.max() else 'no'}",
        "",
        "## Duplication And Leakage Risk",
        f"- Exact duplicate rows: {duplicate_rows}",
        f"- Duplicate symptom patterns: {duplicate_feature_patterns}",
        f"- Conflicting symptom patterns with multiple labels: {conflicting_patterns}",
        "",
        "The very high duplicate-pattern count means random train/test splits can look perfect.",
        "For production, test with real patient cases or a separately collected validation set.",
        "",
        "## Disease Name Cleanup",
        *[f"- `{raw}` -> `{clean}`" for raw, clean in DISEASE_ALIASES.items()],
        "",
        "## Highly Correlated Feature Pairs",
    ]

    if correlation_pairs:
        for left, right, value in correlation_pairs[:30]:
            lines.append(f"- `{left}` / `{right}`: {value:.3f}")
    else:
        lines.append("- No feature pairs above the configured threshold.")

    lines.extend(
        [
            "",
            "## Recommendation Lookup Coverage",
            *lookup_coverage(sorted(labels.unique().tolist())),
            "",
            "## Empty Recommendation Values",
            *empty_recommendation_cells(),
            "",
            "## Recommendation",
            "- Keep the model as a decision-support demo unless validated on independent clinical data.",
            "- Show top predictions and confidence warnings, not only one final disease.",
            "- Ask for more symptoms when fewer than 3 recognised symptoms are provided.",
            "- Flag emergency symptoms and direct users toward urgent care.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved audit report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
