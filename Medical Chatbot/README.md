# NextGen AI Healthcare UI

This folder contains the polished Streamlit interface for the Python-only version
of the project. It combines:

- Symptom-based disease recommendation using the CSV dataset in the sibling
  `Medicine-Recommendation-System-Personalized-Medical-Recommendation-System-with-Machine-Learning`
  folder.
- Medical document chatbot using LangChain, FAISS, HuggingFace embeddings, and
  Groq.

## Setup

```bash
cd "Medical Chatbot"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file if you want to use the chatbot:

```bash
copy .env.example .env
```

Then add your real `GROQ_API_KEY` inside `.env`.

## Run The UI

```bash
streamlit run Bot_UI.py
```

The symptom diagnosis tab works from the existing CSV dataset. The chatbot tab
also needs a FAISS vector database.

## Prepare The Chatbot Knowledge Base

1. Create a `data` folder inside `Medical Chatbot`.
2. Put your medical PDF files in `Medical Chatbot/data/`.
3. Build the FAISS database:

```bash
python Create_Vector_db.py
```

After this, run the Streamlit UI again.

## Run The Recommendation API

The Flask API exposes the recommendation engine for the React UI:

```bash
python flask_api.py
```

Available endpoints:

- `GET /health`
- `GET /api/symptoms`
- `POST /api/recommendations` with JSON body:

```json
{
  "symptoms": ["high_fever", "cough", "fatigue"]
}
```

The response includes the predicted disease, confidence, matched symptoms,
warnings, precautions, medications, diets, exercises, top predictions, and a
structured care plan.

## Notes

This project is educational and should not be used as a replacement for
professional medical advice, diagnosis, or emergency care.
