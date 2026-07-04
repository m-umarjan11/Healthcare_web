# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

This is a monorepo with three independently-run projects that together form one product (an
educational healthcare app: symptom-based disease recommendation + a document-grounded medical
chatbot). There is no root build system tying them together — each has its own dependency
manager and is started as a separate process:

1. **`Medicine-Recommendation-System-Personalized-Medical-Recommendation-System-with-Machine-Learning/`**
   — Python, CSV-backed symptom → disease recommendation engine. No web server of its own; it's
   a library imported by the other two Python entry points.
2. **`Medical Chatbot/`** — Python. Two separate entry points that both wrap the recommendation
   engine and a RAG chatbot: a Streamlit app (`Bot_UI.py`) and a Flask JSON API (`flask_api.py`,
   consumed by `healthcare-ui`).
3. **`healthcare-ui/`** — React + Vite + Tailwind SPA that talks to the Flask API.

`Medical Chatbot` imports the recommendation engine via `sys.path.insert(0, ...)` pointing at the
sibling folder above by its full name — the two directories must stay siblings.

No test suite exists anywhere in this repo (no pytest/jest config, no test files). Don't assume one.

## Commands

### Medicine-Recommendation-System (`Medicine-Recommendation-System-Personalized-Medical-Recommendation-System-with-Machine-Learning/`)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt          # pandas, scikit-learn, joblib

python predict_cli.py "itching, skin rash, nodal skin eruptions"   # CLI prediction, fuzzy-matches symptom names
python train_model.py                    # trains/compares SVC, RandomForest, GradientBoosting, KNN,
                                          # MultinomialNB, MLP; saves the best to models/best_model.joblib
python data_audit.py                     # writes models/data_audit_report.md (class balance, duplicates,
                                          # correlated symptoms, lookup coverage)
```

`models/` is gitignored — `train_model.py` must be run locally to produce `best_model.joblib`
before `MedicalRecommendationEngine` will use a pre-trained model (it otherwise trains one
in-memory on first use, which is slower).

### Medical Chatbot (`Medical Chatbot/`)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt          # or: pipenv install (Pipfile also present)
copy .env.example .env                   # then set GROQ_API_KEY

streamlit run Bot_UI.py                  # Streamlit UI: symptom diagnosis + chatbot tabs
python flask_api.py                      # Flask API for healthcare-ui (default port 5000)
python Create_Vector_db.py               # (re)builds the FAISS index from PDFs in Medical Chatbot/data/
python Connection_LLM_Vectordb.py        # ad hoc CLI query against the RAG chain
```

`Medical Chatbot/data/` (source PDFs) and `Medical Chatbot/vectorstore/` (FAISS index) are
gitignored generated artifacts. To rebuild the knowledge base from scratch: delete
`vectorstore/db_faiss/`, drop PDFs into `data/`, then rerun `Create_Vector_db.py`.

Flask API env vars: `GROQ_API_KEY` (required for chatbot routes), `GROQ_MODEL_NAME` (defaults to
`llama-3.1-8b-instant`), `ALLOWED_ORIGIN`/`ALLOWED_ORIGINS` (comma-separated CORS allowlist;
`localhost`/`127.0.0.1` on any port is always allowed), `PORT` (default 5000), `FLASK_DEBUG`.

### healthcare-ui (`healthcare-ui/`)

```bash
npm install
npm run dev          # vite dev server
npm run build         # production build to dist/
npm run lint          # eslint .
```

Points at the Flask API via `VITE_API_BASE_URL` (default `http://127.0.0.1:5000`).

## Architecture

### Recommendation engine (`recommendation_engine.py`)

`MedicalRecommendationEngine` is the single class both Python entry points use. Key behavior to
know before touching it:

- Loads `Training.csv` (binary symptom matrix + `prognosis` label), canonicalizes known-misspelled
  disease names via `DISEASE_ALIASES`, and drops duplicate rows.
- `model` property: tries to load `models/best_model.joblib` first (validated against the current
  `symptom_columns` order — mismatched saved models are ignored); otherwise trains a
  `RandomForestClassifier` in-memory on an *augmented* training set built by
  `_model_training_df()`, which generates synthetic partial-symptom rows from `combinations()` of
  each row's active symptoms (capped per row) and drops any generated pattern that would map to
  more than one disease.
- `predict()` combines two signals: the ML model's `predict_proba` ranking, and a template-overlap
  score (`_template_predictions`) that scores precision/recall/Jaccard overlap between the input
  symptoms and each disease's known symptom patterns. Template predictions win when non-empty;
  the model is the fallback.
- Symptom/disease name matching is fuzzy (`difflib`) and normalizes by stripping non-alphanumerics
  (`_normalise_key`), so lookups are resilient to casing/spacing/typo differences against the CSV
  lookup tables (`description.csv`, `precautions_df.csv`, `medications.csv`, `diets.csv`,
  `workout_df.csv`, `Symptom-severity.csv`).
- `EMERGENCY_SYMPTOMS` triggers a distinct safety note; `_prediction_warnings` flags low symptom
  counts, unrecognized symptoms, low confidence, and close top-2 predictions — these feed the
  `reliability` label (`high`/`moderate`/`low`) surfaced in both UIs.

`train_model.py` is a separate, standalone script (not imported by the engine) used to produce
`models/best_model.joblib` offline.

### RAG chatbot core (`Medical Chatbot/medical_rag.py`)

This module is the single source of truth for the chatbot pipeline and is shared by both
`Bot_UI.py` and `flask_api.py`:

- `_get_pipeline()` (`@lru_cache`) builds and caches the FAISS retriever (MMR search,
  `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`), the `ChatGroq` LLM
  client, and the prompt — expensive (~15-20s, dominated by the embedding model load) and done
  once per process.
- `prewarm()` kicks off `_get_pipeline()` on a background thread, guarded to run once per
  process. Both `Bot_UI.py` (module-level, on Streamlit script load) and `flask_api.py`
  (module-level, on import) call this so the cold-start cost overlaps with the user browsing
  instead of blocking the first chat message.
- `build_qa_chain()` — blocking `invoke()`-style chain (used by `Connection_LLM_Vectordb.py` and
  the non-streaming `/api/chatbot` and `/ask` Flask routes).
- `stream_answer(question)` — retrieves source documents eagerly (fast, tens of ms), then returns
  `(source_documents, token_generator)`; the generator yields LLM tokens via `ChatGroq.stream()`.
  This is what powers real-time streaming in both UIs.
- The embedding model auto-detects whether it's already been downloaded to the local
  `huggingface_hub` cache and, if so, sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` so it never
  makes a network call to check for updates — the chatbot must keep working with no internet once
  the model and FAISS index exist locally.
- The prompt (`MEDICAL_RAG_PROMPT`) assumes its output is rendered as **plain text, not Markdown**
  (the React chat UI does not parse Markdown) — it must never be changed to use `**bold**`, `##`,
  or numbered list markers; it should read as natural prose paragraphs, skipping any section
  (urgency, next steps, etc.) that has nothing genuine to add rather than padding with boilerplate.

### Two chatbot front-ends, one backend flow

- **Streamlit (`Bot_UI.py`)**: single process, calls `medical_rag.stream_answer()` directly
  in-process and renders it with `st.write_stream()`. The recommendation engine is loaded via
  `st.cache_resource`/`st.cache_data`.
- **Flask + React**: `flask_api.py` exposes `POST /api/chatbot/stream`, which streams
  newline-delimited JSON events (`{"type": "token", "content": ...}`, then a final
  `{"type": "done", "sources": [...]}`, or `{"type": "error", "message": ...}`) built directly on
  top of `stream_answer()`. `healthcare-ui/src/App.jsx`'s `askChatbot()` reads the response body
  via `ReadableStream`/`TextDecoder`, appending tokens to the last assistant message as they
  arrive. The older `POST /api/chatbot` (blocking, full JSON) and `GET /ask` routes still exist for
  non-streaming callers.
- Both front ends get sources back as `(filename, page)` pairs derived from FAISS document
  metadata — `Path(...).name` is used to strip the `data/` prefix before display.

### Recommendation flow (Flask ↔ React)

`flask_api.py`'s `/api/recommendations` wraps `MedicalRecommendationEngine.predict()` and adds a
`build_care_plan()` step that turns the raw `PredictionResult` into four UI-ready sections
(review the prediction, follow precautions, support recovery, know when to seek care) — this
shaping logic lives only in `flask_api.py`, not in the engine itself. `healthcare-ui`'s diagnosis
screen posts selected symptom values (matching the engine's raw column names, from
`GET /api/symptoms`) and renders the response on `ResultsScreen`/care-plan screens.

### CORS

`flask_api.py` hand-rolls CORS in `add_cors_headers` (an `after_request` hook) rather than using
`flask-cors`: it echoes back the request's `Origin` header only if it's in the configured allowlist
or matches `localhost`/`127.0.0.1` on any port; otherwise it falls back to the first configured
allowed origin. Extend `ALLOWED_ORIGIN(S)` rather than adding a new CORS mechanism.


claude --resume a2296a9a-f848-4a43-b672-2e17a35ff962