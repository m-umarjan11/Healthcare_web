from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RECOMMENDATION_DIR = (
    PROJECT_DIR
    / "Medicine-Recommendation-System-Personalized-Medical-Recommendation-System-with-Machine-Learning"
)
sys.path.insert(0, str(RECOMMENDATION_DIR))

from medical_rag import prewarm

prewarm()

from recommendation_engine import MedicalRecommendationEngine, PredictionResult


load_dotenv(find_dotenv())

app = Flask(__name__)

DEFAULT_ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}
LOCAL_DEV_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1):\d+$")


def allowed_origins() -> set[str]:
    configured = os.getenv("ALLOWED_ORIGIN") or os.getenv("ALLOWED_ORIGINS")
    if not configured:
        return DEFAULT_ALLOWED_ORIGINS
    return {origin.strip() for origin in configured.split(",") if origin.strip()}


def is_allowed_origin(origin: str | None) -> bool:
    if not origin:
        return False
    if origin in allowed_origins():
        return True
    return bool(LOCAL_DEV_ORIGIN_RE.match(origin))


@app.after_request
def add_cors_headers(response):
    request_origin = request.headers.get("Origin")
    if is_allowed_origin(request_origin):
        response.headers["Access-Control-Allow-Origin"] = request_origin
    else:
        response.headers["Access-Control-Allow-Origin"] = next(iter(allowed_origins()))
    response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@lru_cache(maxsize=1)
def get_recommendation_engine() -> MedicalRecommendationEngine:
    return MedicalRecommendationEngine(RECOMMENDATION_DIR)


@lru_cache(maxsize=1)
def get_qa_chain():
    from medical_rag import build_qa_chain

    return build_qa_chain()


def serialize_prediction(result: PredictionResult) -> dict[str, object]:
    return {
        "disease": result.disease,
        "confidence": result.confidence,
        "confidence_percent": round(result.confidence * 100, 1)
        if result.confidence is not None
        else None,
        "reliability": result.reliability,
        "description": result.description,
        "precautions": result.precautions,
        "medications": result.medications,
        "diets": result.diets,
        "exercises": result.workouts,
        "symptom_severity": result.symptom_severity,
        "warnings": result.warnings,
        "safety_notes": result.safety_notes,
        "unknown_symptoms": result.unknown_symptoms,
        "matched_symptoms": [
            {
                "input": match.original,
                "matched": match.matched,
                "confidence": match.confidence,
            }
            for match in result.matched_symptoms
        ],
        "top_predictions": [
            {
                "disease": disease,
                "probability": probability,
                "probability_percent": round(probability * 100, 1),
            }
            for disease, probability in result.top_predictions
        ],
        "care_plan": build_care_plan(result),
    }


def build_care_plan(result: PredictionResult) -> list[dict[str, object]]:
    return [
        {
            "title": "Review the prediction",
            "items": [
                f"Likely condition: {result.disease}",
                f"Reliability: {result.reliability.title()}",
                "Compare with the alternate predictions before acting.",
            ],
        },
        {
            "title": "Follow precautions",
            "items": result.precautions[:4]
            or ["Track symptoms and avoid self-treatment if symptoms worsen."],
        },
        {
            "title": "Support recovery",
            "items": [
                *(result.diets[:3] or ["Maintain hydration and balanced meals."]),
                *(result.workouts[:2] or ["Rest and resume activity gradually."]),
            ],
        },
        {
            "title": "Know when to seek care",
            "items": result.safety_notes
            + (result.warnings or ["Consult a qualified clinician for diagnosis."]),
        },
    ]


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/api/symptoms", methods=["GET"])
def symptom_options():
    engine = get_recommendation_engine()
    return jsonify(
        {
            "symptoms": engine.symptom_options(),
            "total": len(engine.symptom_columns),
        }
    )


@app.route("/api/recommendations", methods=["POST", "OPTIONS"])
def recommendations():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    symptoms = payload.get("symptoms", [])
    if isinstance(symptoms, str):
        symptoms = [part.strip() for part in symptoms.split(",")]
    symptoms = [str(symptom).strip() for symptom in symptoms if str(symptom).strip()]

    if not symptoms:
        return jsonify({"error": "Provide at least one symptom."}), 400

    try:
        result = get_recommendation_engine().predict(symptoms)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Recommendation failed: {exc}"}), 500

    return jsonify(serialize_prediction(result))


def _unique_by_source(source_documents: list) -> list:
    """Keep only the first document seen per source PDF, so a single file
    that matched several pages appears once in the sources list instead of
    once per page."""
    seen = set()
    unique_documents = []
    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown source")
        if source in seen:
            continue
        seen.add(source)
        unique_documents.append(doc)
    return unique_documents


def serialize_chatbot_response(user_query: str, response: dict) -> dict[str, object]:
    return {
        "query": user_query,
        "answer": response.get("result", "No answer found."),
        "sources": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown source"),
                "page": doc.metadata.get("page"),
                "metadata": doc.metadata,
            }
            for doc in _unique_by_source(response.get("source_documents", []))
        ],
    }


@app.route("/api/chatbot", methods=["POST", "OPTIONS"])
def chatbot():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    user_query = str(payload.get("question") or payload.get("query") or "").strip()
    if not user_query:
        return jsonify({"error": "Provide a medical question."}), 400

    try:
        response = get_qa_chain().invoke({"query": user_query})
        return jsonify(serialize_chatbot_response(user_query, response))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chatbot/stream", methods=["POST", "OPTIONS"])
def chatbot_stream():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    user_query = str(payload.get("question") or payload.get("query") or "").strip()
    if not user_query:
        return jsonify({"error": "Provide a medical question."}), 400

    def generate():
        from medical_rag import generate_followups, stream_answer

        try:
            source_documents, token_stream = stream_answer(user_query)
            answer_parts = []
            for chunk in token_stream:
                answer_parts.append(chunk)
                yield json.dumps({"type": "token", "content": chunk}) + "\n"

            sources = [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown source"),
                    "page": doc.metadata.get("page"),
                }
                for doc in _unique_by_source(source_documents)
            ]
            yield json.dumps({"type": "done", "sources": sources}) + "\n"

            try:
                suggestions = generate_followups(user_query, "".join(answer_parts))
                yield json.dumps({"type": "suggestions", "questions": suggestions}) + "\n"
            except Exception:
                pass  # follow-up suggestions are a nice-to-have; never fail the response over them
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
    )


@app.route("/ask", methods=["GET"])
def ask_question():
    user_query = request.args.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Please provide a 'query' parameter."}), 400

    try:
        response = get_qa_chain().invoke({"query": user_query})
        data = serialize_chatbot_response(user_query, response)
        data["result"] = data["answer"]
        data["source_documents"] = [
            {"content": source["content"], "metadata": source["metadata"]}
            for source in data["sources"]
        ]
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        port=int(os.getenv("PORT", "5000")),
        use_reloader=False,
    )
