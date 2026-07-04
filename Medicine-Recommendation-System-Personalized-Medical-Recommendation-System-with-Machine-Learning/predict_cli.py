import argparse

from recommendation_engine import MedicalRecommendationEngine


def format_list(values: list[str]) -> str:
    if not values:
        return "No data available."
    return "\n".join(f"- {value}" for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a likely disease from comma-separated symptoms."
    )
    parser.add_argument(
        "symptoms",
        help='Comma-separated symptoms, for example: "itching, skin rash, fatigue"',
    )
    args = parser.parse_args()

    symptoms = [symptom.strip() for symptom in args.symptoms.split(",")]
    engine = MedicalRecommendationEngine()
    result = engine.predict(symptoms)

    print("\nPredicted Disease")
    print("=================")
    if result.confidence is None:
        print(result.disease)
    else:
        print(f"{result.disease} ({result.confidence * 100:.1f}% confidence)")
    print(f"Reliability: {result.reliability}")

    if result.warnings:
        print("\nPrediction Warnings")
        print("===================")
        print(format_list(result.warnings))

    if result.safety_notes:
        print("\nSafety Notes")
        print("============")
        print(format_list(result.safety_notes))

    if result.unknown_symptoms:
        print("\nUnrecognised Symptoms")
        print("=====================")
        print(", ".join(result.unknown_symptoms))

    print("\nMatched Symptoms")
    print("================")
    for match in result.matched_symptoms:
        if match.matched:
            print(f"- {match.original} -> {match.matched} ({match.confidence:.2f})")

    print("\nDescription")
    print("===========")
    print(result.description or "No description available.")

    print("\nPrecautions")
    print("===========")
    print(format_list(result.precautions))

    print("\nMedications")
    print("===========")
    print(format_list(result.medications))

    print("\nDiet")
    print("====")
    print(format_list(result.diets))

    print("\nLifestyle / Workout")
    print("===================")
    print(format_list(result.workouts[:8]))

    print("\nTop Predictions")
    print("===============")
    for disease, probability in result.top_predictions:
        print(f"- {disease}: {probability * 100:.1f}%")


if __name__ == "__main__":
    main()
