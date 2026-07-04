import ast
import difflib
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RECOMMENDATION_DIR = (
    PROJECT_DIR
    / "Medicine-Recommendation-System-Personalized-Medical-Recommendation-System-with-Machine-Learning"
)
sys.path.insert(0, str(RECOMMENDATION_DIR))

from recommendation_engine import MedicalRecommendationEngine
from medical_rag import DB_FAISS_PATH, build_qa_chain, prewarm, stream_answer

prewarm()

st.set_page_config(
    page_title="NextGen AI Healthcare",
    page_icon="health",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #f7fbff 0%, #eef4ff 42%, #f8fff8 100%);
    }
    .hero {
        padding: 2rem;
        border-radius: 28px;
        color: white;
        background:
            radial-gradient(circle at top right, rgba(255,255,255,.28), transparent 34%),
            linear-gradient(135deg, #2347c5 0%, #0aa3a3 100%);
        box-shadow: 0 20px 45px rgba(35, 71, 197, .18);
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        margin-bottom: .35rem;
    }
    .hero p {
        font-size: 1.05rem;
        max-width: 780px;
        opacity: .94;
    }
    .metric-card {
        background: rgba(255,255,255,.82);
        border: 1px solid rgba(30, 64, 175, .08);
        border-radius: 22px;
        padding: 1.2rem;
        box-shadow: 0 14px 35px rgba(15, 23, 42, .07);
    }
    .section-card {
        background: rgba(255,255,255,.88);
        border: 1px solid rgba(15, 23, 42, .08);
        border-radius: 24px;
        padding: 1.3rem;
        box-shadow: 0 16px 42px rgba(15, 23, 42, .08);
    }
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        border-radius: 16px;
        padding: .95rem 1rem;
        margin-top: .8rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 18px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, .06);
    }
</style>
"""


def prettify(value: str) -> str:
    return value.replace("_", " ").strip().title()


def parse_list_cell(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


@st.cache_data(show_spinner=False)
def load_recommendation_data():
    engine = MedicalRecommendationEngine(RECOMMENDATION_DIR)
    return engine.training_df, engine.symptom_columns, engine.model


@st.cache_resource(show_spinner=False)
def load_recommendation_engine():
    return MedicalRecommendationEngine(RECOMMENDATION_DIR)


@st.cache_data(show_spinner=False)
def load_lookup_table(filename: str) -> pd.DataFrame:
    path = RECOMMENDATION_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def find_column(df: pd.DataFrame, wanted: str) -> str | None:
    wanted_key = normalize_key(wanted)
    for column in df.columns:
        if normalize_key(column) == wanted_key:
            return column
    return None


def find_disease_row(df: pd.DataFrame, disease: str):
    disease_column = find_column(df, "Disease")
    if df.empty or disease_column is None:
        return None

    disease_key = normalize_key(disease)
    keys = df[disease_column].astype(str).map(normalize_key)
    exact_match = df[keys == disease_key]
    if not exact_match.empty:
        return exact_match.iloc[0]

    close_matches = difflib.get_close_matches(disease_key, keys.tolist(), n=1, cutoff=0.84)
    if close_matches:
        return df[keys == close_matches[0]].iloc[0]
    return None


def get_single_value(df: pd.DataFrame, disease: str, value_column: str) -> str:
    value_column = find_column(df, value_column)
    if df.empty or value_column is None:
        return ""
    row = find_disease_row(df, disease)
    if row is None or pd.isna(row[value_column]):
        return ""
    return str(row[value_column])


def get_list_value(df: pd.DataFrame, disease: str, value_column: str) -> list[str]:
    value = get_single_value(df, disease, value_column)
    return parse_list_cell(value)


def get_precautions(disease: str) -> list[str]:
    df = load_lookup_table("precautions_df.csv")
    row = find_disease_row(df, disease)
    if row is None:
        return []
    return [
        str(row[col]).strip()
        for col in df.columns
        if col.lower().startswith("precaution") and pd.notna(row[col]) and str(row[col]).strip()
    ]


def get_workouts(disease: str) -> list[str]:
    df = load_lookup_table("workout_df.csv")
    if df.empty:
        return []
    disease_column = find_column(df, "disease")
    workout_column = find_column(df, "workout")
    if disease_column is None or workout_column is None:
        return []

    disease_key = normalize_key(disease)
    keys = df[disease_column].astype(str).map(normalize_key)
    matched = df[keys == disease_key]
    if matched.empty:
        close_matches = difflib.get_close_matches(disease_key, keys.unique().tolist(), n=1, cutoff=0.84)
        if close_matches:
            matched = df[keys == close_matches[0]]

    return [
        str(value).strip()
        for value in matched[workout_column].dropna().tolist()
        if str(value).strip()
    ]


def predict_disease(selected_symptoms: list[str]):
    _, symptom_columns, model = load_recommendation_data()
    row = pd.DataFrame([[0] * len(symptom_columns)], columns=symptom_columns)
    for symptom in selected_symptoms:
        if symptom in row.columns:
            row.at[0, symptom] = 1

    prediction = model.predict(row)[0]
    probabilities = []
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(row)[0]
        probabilities = sorted(
            zip(model.classes_, probas),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
    return prediction, probabilities


@st.cache_resource(show_spinner=False)
def check_qa_chain_ready() -> tuple[bool, str]:
    try:
        build_qa_chain()
        return True, ""
    except RuntimeError as exc:
        return False, str(exc)


def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>NextGen AI Healthcare</h1>
            <p>
                A focused AI workspace for symptom-based disease recommendations
                and document-grounded medical question answering.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    training_df, symptom_columns, _ = load_recommendation_data()

    st.sidebar.title("Project Status")
    st.sidebar.success("Recommendation system ready")
    if os.getenv("GROQ_API_KEY"):
        st.sidebar.success("Groq API key detected")
    else:
        st.sidebar.warning("Groq API key missing")

    if DB_FAISS_PATH.exists():
        st.sidebar.success("FAISS vector DB found")
    else:
        st.sidebar.warning("FAISS vector DB missing")

    st.sidebar.divider()
    st.sidebar.metric("Diseases", training_df["prognosis"].nunique())
    st.sidebar.metric("Symptoms", len(symptom_columns))
    st.sidebar.metric("Training Rows", len(training_df))

    st.sidebar.markdown(
        """
        <div class="disclaimer">
            This tool is for educational support only. For urgent symptoms,
            severe pain, breathing difficulty, or worsening condition, contact
            a qualified medical professional immediately.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diagnosis_tab():
    training_df, symptom_columns, _ = load_recommendation_data()
    symptom_options = {prettify(symptom): symptom for symptom in symptom_columns}

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Symptom Checker")
        selected_labels = st.multiselect(
            "Select the symptoms you are experiencing",
            options=list(symptom_options.keys()),
            placeholder="Search symptoms such as fever, cough, headache...",
        )
        selected_symptoms = [symptom_options[label] for label in selected_labels]

        st.caption("Tip: choose at least 3 symptoms for a more meaningful prediction.")
        diagnose = st.button("Analyze Symptoms", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Dataset Overview")
        st.write(
            f"The model is trained from `{len(training_df)}` rows, "
            f"`{len(symptom_columns)}` symptoms, and "
            f"`{training_df['prognosis'].nunique()}` disease classes."
        )
        st.markdown("**Common disease classes in this dataset:**")
        st.write(", ".join(sorted(training_df["prognosis"].unique())[:12]) + "...")
        st.markdown("</div>", unsafe_allow_html=True)

    if diagnose:
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
            return

        result = load_recommendation_engine().predict(selected_symptoms)

        st.divider()
        st.markdown("### Result")
        result_col, prob_col = st.columns([1.1, 0.9], gap="large")

        with result_col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.caption("Most likely condition")
            st.markdown(f"## {result.disease}")
            st.caption(f"Reliability: {result.reliability.title()}")
            if result.description:
                st.write(result.description)
            st.markdown("</div>", unsafe_allow_html=True)

        if result.warnings:
            st.warning(" ".join(result.warnings))

        if result.safety_notes:
            st.info(" ".join(result.safety_notes))

        with prob_col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.caption("Top model matches")
            for label, score in result.top_predictions:
                st.progress(float(score), text=f"{label}: {score * 100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        rec_cols = st.columns(4)
        sections = [
            ("Precautions", result.precautions),
            ("Medications", result.medications),
            ("Diet", result.diets),
            ("Lifestyle", result.workouts),
        ]
        for col, (title, values) in zip(rec_cols, sections):
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                if values:
                    for value in values[:6]:
                        st.write(f"- {value}")
                else:
                    st.caption("No data available.")
                st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Matched Symptoms"):
            for match in result.matched_symptoms:
                if match.matched:
                    st.write(
                        f"- {match.original} -> {prettify(match.matched)} "
                        f"({match.confidence:.2f})"
                    )
            if result.unknown_symptoms:
                st.warning("Unknown symptoms: " + ", ".join(result.unknown_symptoms))


def render_chatbot_tab():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Medical Chatbot")
    st.write(
        "Ask questions against your medical PDF knowledge base. "
        "Answers are retrieved from FAISS and generated with Groq."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask a medical question from your uploaded knowledge base...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    ready, error = check_qa_chain_ready()
    if not ready:
        st.chat_message("assistant").warning(error)
        st.session_state.messages.append({"role": "assistant", "content": error})
        return

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching medical knowledge base..."):
                source_documents, token_stream = stream_answer(prompt)
            answer = st.write_stream(token_stream)
        except Exception as exc:
            error_message = f"Sorry, something went wrong answering that question: {exc}"
            st.warning(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            return

        seen_sources = set()
        unique_source_documents = []
        for doc in source_documents:
            source = doc.metadata.get("source", "Unknown source")
            if source in seen_sources:
                continue
            seen_sources.add(source)
            unique_source_documents.append(doc)

        source_lines = [
            f"- {Path(doc.metadata.get('source', 'Unknown source')).name}, "
            f"page {doc.metadata.get('page', 'N/A')}"
            for doc in unique_source_documents
        ]
        final_answer = answer
        if source_lines:
            sources_block = "\n\n**Sources**\n" + "\n".join(source_lines)
            st.markdown(sources_block)
            final_answer += sources_block

    st.session_state.messages.append({"role": "assistant", "content": final_answer})


def render_about_tab():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("About This AI Workspace")
    st.write(
        "This cleaned project keeps only the symptom recommendation system and "
        "the medical chatbot. The Flutter mobile app, Node backend, and unrelated "
        "documents have been removed from the main workspace."
    )
    st.markdown(
        """
        **Run the UI**

        ```bash
        cd "Medical Chatbot"
        streamlit run Bot_UI.py
        ```

        **Prepare chatbot knowledge base**

        ```bash
        # Put PDF files in Medical Chatbot/data/
        python Create_Vector_db.py
        ```
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    render_header()
    render_sidebar()

    diagnosis_tab, chatbot_tab, about_tab = st.tabs(
        ["Symptom Diagnosis", "Medical Chatbot", "About"]
    )
    with diagnosis_tab:
        render_diagnosis_tab()
    with chatbot_tab:
        render_chatbot_tab()
    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
