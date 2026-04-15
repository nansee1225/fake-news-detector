# ==============================================
# ADVANCED FAKE NEWS DETECTOR (MULTI-FEATURE AI PROJECT)
# ==============================================
# Features:
# 1. Text-based fake news classification (ML + NLP)
# 2. TF-IDF + Logistic Regression model
# 3. Confidence score
# 4. Explainability (top important words)
# 5. URL news extraction
# 6. Simple GUI using Streamlit
# ==============================================

# INSTALL DEPENDENCIES:
# pip install pandas numpy scikit-learn streamlit requests beautifulsoup4

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================
# Use any dataset with 'text' and 'label' columns (label: 0=real, 1=fake)

def load_data():
    try:
        df = pd.read_csv("news.csv")
    except:
        # fallback small dataset
        data = {
            'text': [
                "Government launches new scheme for farmers",
                "Aliens landed in India yesterday",
                "Stock market reaches all time high",
                "Celebrity cloned secretly in lab"
            ],
            'label': [0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
    return df

# =========================
# PREPROCESS + TRAIN MODEL
# =========================

def train_model(df):
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, acc

# =========================
# EXPLAINABILITY FUNCTION
# =========================

def explain_prediction(text, model, vectorizer):
    feature_names = np.array(vectorizer.get_feature_names_out())
    vector = vectorizer.transform([text])

    coefs = model.coef_[0]
    top_indices = np.argsort(vector.toarray()[0] * coefs)[-5:]

    return feature_names[top_indices]

# =========================
# URL NEWS EXTRACTION
# =========================

def fetch_news_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except:
        return "Could not fetch content"

# =========================
# STREAMLIT UI
# =========================

def main():
    st.title("📰 Advanced Fake News Detector")

    df = load_data()
    model, vectorizer, acc = train_model(df)

    st.sidebar.write(f"Model Accuracy: {acc:.2f}")

    option = st.radio("Choose Input Type:", ("Text", "URL"))

    if option == "Text":
        user_input = st.text_area("Enter News Text:")
    else:
        url = st.text_input("Enter News URL:")
        user_input = fetch_news_from_url(url) if url else ""

    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter valid input")
        else:
            vector = vectorizer.transform([user_input])
            prediction = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]

            label = "FAKE ❌" if prediction == 1 else "REAL ✅"
            confidence = max(prob)

            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}")

            keywords = explain_prediction(user_input, model, vectorizer)
            st.write("Top Influencing Words:", keywords)

            st.text_area("Extracted/Entered Text", user_input, height=200)

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    main()
