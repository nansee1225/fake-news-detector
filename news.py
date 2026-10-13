# ======================================================
# SCROLL-BASED FAKE NEWS DETECTOR (NO NAVBAR SWITCHING)
# Single Page App with Sections (Smooth Scroll Feel)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fake News Detector Pro", layout="wide")

USER_DB = "users.json"

# ---------------- INIT ----------------
if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

# ---------------- STYLE ----------------
st.markdown("""
<style>
.big-title {font-size: 42px; font-weight: bold; color: #22c55e; margin-top: 30px;}
.section {margin-top: 60px; padding: 20px; background: #1f2937; border-radius: 15px;}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
dataset_path = Path(__file__).resolve().with_name("news_dataset.csv")
news_data = pd.read_csv(dataset_path)
required_columns = {"text", "label"}
if not required_columns.issubset(news_data.columns):
    raise ValueError("news_dataset.csv must contain 'text' and 'label' columns.")

texts = news_data["text"].fillna("").astype(str).str.strip()
labels = news_data["label"].fillna("").astype(str).str.strip().str.upper()
if texts.eq("").any():
    raise ValueError("news_dataset.csv contains an empty news text.")
if not labels.isin({"REAL", "FAKE"}).all():
    raise ValueError("news_dataset.csv labels must be either 'REAL' or 'FAKE'.")
if labels.nunique() < 2:
    raise ValueError("news_dataset.csv must contain examples for both labels.")

labels = labels.map({"REAL": 1, "FAKE": 0})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

# ---------------- AUTH ----------------
def load_users():
    with open(USER_DB, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

# ---------------- LOGIN ----------------
def auth_page():
    st.markdown("<div class='big-title'>🔐 Login / Signup</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Signup"])
    users = load_users()

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in users and users[u] == p:
                st.session_state.logged_in = True
                st.session_state.user = u
                st.success("Login successful")
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("Create Username")
        p = st.text_input("Create Password", type="password")
        if st.button("Signup"):
            if u in users:
                st.warning("User exists")
            else:
                users[u] = p
                save_users(users)
                st.success("Account created")

# ---------------- MAIN PAGE (SCROLL) ----------------
def main_page():

    st.title(f"👋 Welcome, {st.session_state.user}")

    # -------- HOME --------
    st.markdown("<div class='big-title'>📰 Home</div>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1504711434969-e33886168f5c", use_container_width=True)
    st.write("Scroll down to explore features ↓")

    # -------- DASHBOARD --------
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>📊 Dashboard</div>", unsafe_allow_html=True)

    real = random.randint(50, 100)
    fake = random.randint(20, 60)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", real + fake)
    col2.metric("Real", real)
    col3.metric("Fake", fake)

    df = pd.DataFrame({"Type": ["Real", "Fake"], "Count": [real, fake]})
    fig, ax = plt.subplots()
    ax.bar(df["Type"], df["Count"])
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- DETECTOR --------
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>🔍 Detector</div>", unsafe_allow_html=True)

    text = st.text_area("Enter news text")
    if st.button("Analyze"):
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("REAL news")
        else:
            st.error("FAKE news")
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- ANALYTICS --------
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>📈 Analytics</div>", unsafe_allow_html=True)

    vals = [random.randint(10, 100) for _ in range(7)]
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    fig, ax = plt.subplots()
    ax.plot(days, vals)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------- LOGOUT --------
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# ---------------- RUN ----------------
if not st.session_state.logged_in:
    auth_page()
else:
    main_page()
