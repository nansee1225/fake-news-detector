from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DATASET_PATH = Path(__file__).resolve().with_name("news_dataset.csv")

st.set_page_config(
    page_title="Newscheck | Misinformation Analyzer",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #f3f5f2;
        --muted: #9aa59f;
        --line: rgba(206, 224, 214, 0.12);
        --green: #a8f074;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(71, 118, 81, .17), transparent 34%),
            #0b100f;
        color: var(--ink);
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: -0.035em;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }
    .eyebrow {
        color: var(--green);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin-bottom: .8rem;
    }
    .hero-title {
        color: var(--ink);
        font-family: 'Segoe UI', sans-serif;
        font-size: clamp(2.4rem, 6vw, 4.6rem);
        font-weight: 800;
        letter-spacing: -.065em;
        line-height: 1.04;
        margin: 0;
    }
    .hero-title span { color: var(--green); }
    .hero-copy {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.7;
        max-width: 610px;
        margin: 1rem 0 2.3rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(22, 32, 29, .98), rgba(16, 24, 22, .98));
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.5rem;
    }
    .panel-label {
        color: var(--muted);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .stTextArea textarea {
        background: #0c1311;
        border: 1px solid var(--line);
        border-radius: 12px;
        color: var(--ink);
        line-height: 1.65;
    }
    .stTextArea textarea:focus {
        border-color: rgba(168, 240, 116, .55);
        box-shadow: 0 0 0 1px rgba(168, 240, 116, .2);
    }
    .stButton > button, .stFormSubmitButton > button {
        background: var(--green);
        border: 0;
        border-radius: 10px;
        color: #10170d;
        font-weight: 700;
        min-height: 2.8rem;
        transition: transform .15s ease, filter .15s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: var(--green);
        color: #10170d;
        filter: brightness(1.06);
        transform: translateY(-1px);
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, .025);
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .note {
        border-left: 2px solid var(--green);
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.6;
        margin-top: 1.2rem;
        padding-left: .85rem;
    }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATASET_PATH)
    required_columns = {"text", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError("news_dataset.csv must contain 'text' and 'label' columns.")

    data = data.loc[:, ["text", "label"]].copy()
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data["label"] = data["label"].fillna("").astype(str).str.strip().str.upper()

    if data["text"].eq("").any():
        raise ValueError("news_dataset.csv contains an empty news text.")
    if not data["label"].isin({"REAL", "FAKE"}).all():
        raise ValueError("news_dataset.csv labels must be either 'REAL' or 'FAKE'.")
    if data["label"].nunique() != 2:
        raise ValueError("news_dataset.csv must contain examples for both labels.")

    return data


@st.cache_resource
def train_model() -> tuple[LogisticRegression, TfidfVectorizer]:
    data = load_dataset()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    features = vectorizer.fit_transform(data["text"])
    labels = data["label"].map({"REAL": 1, "FAKE": 0})

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(features, labels)
    return model, vectorizer


dataset = load_dataset()
model, vectorizer = train_model()

st.markdown(
    """
    <div class="eyebrow">A machine-learning news screening tool</div>
    <h1 class="hero-title">Read between<br>the <span>headlines.</span></h1>
    <p class="hero-copy">
        Check a news excerpt against a text-classification model trained on a
        small sample dataset. Use its result as a starting point—not a substitute
        for trusted reporting and source verification.
    </p>
    """,
    unsafe_allow_html=True,
)


def render_analyzer() -> None:
    st.markdown('<div class="panel-label">Article analysis</div>', unsafe_allow_html=True)
    st.subheader("What would you like to check?")

    with st.form("news_analysis"):
        article_text = st.text_area(
            "Paste a headline or article excerpt",
            height=220,
            max_chars=10000,
            placeholder="Paste the text you want to analyze...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze article  →", width="stretch")

    if submitted:
        article_text = article_text.strip()
        if not article_text:
            st.warning("Paste a headline or article excerpt before analyzing.")
        else:
            features = vectorizer.transform([article_text])
            prediction = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]
            confidence = float(probabilities[list(model.classes_).index(prediction)])

            st.divider()
            if prediction == 1:
                st.success("The model classifies this text as likely REAL.")
            else:
                st.error("The model classifies this text as likely FAKE.")
            st.metric("Model confidence", f"{confidence:.0%}")
            st.caption(
                "Confidence is the model's score for this label, not a measure "
                "of whether the article is factually correct."
            )


def render_model_details() -> None:
    st.markdown('<div class="panel-label">About this model</div>', unsafe_allow_html=True)
    st.subheader("A small, transparent baseline")
    st.write(
        "The classifier converts text into TF-IDF features and uses Logistic "
        "Regression to choose between the two labels."
    )

    real_count = int(dataset["label"].eq("REAL").sum())
    fake_count = int(dataset["label"].eq("FAKE").sum())
    st.markdown(f"**{len(dataset)} examples** in the training set")
    st.caption(f"{real_count} REAL examples  ·  {fake_count} FAKE examples")
    st.markdown(
        '<div class="note">The bundled examples are synthetic and intentionally '
        'small. This demo is not validated for real-world fact-checking.</div>',
        unsafe_allow_html=True,
    )


analyzer, details = st.columns([1.65, 1], gap="large")
with analyzer:
    with st.container(border=True):
        render_analyzer()

with details:
    with st.container(border=True):
        render_model_details()

st.write("")
with st.expander("View the sample training data"):
    st.dataframe(dataset, width="stretch", hide_index=True)
