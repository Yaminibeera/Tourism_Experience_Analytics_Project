"""
Tourism Experience Analytics — Streamlit App
=============================================
Lets a user enter their profile/trip details and get:
  - A predicted attraction rating (regression)
  - A predicted visit mode (classification)
  - Personalized attraction recommendations
Plus an EDA tab with key visualizations.

Run with:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
from train_models import recommend_for_user  # noqa: E402

DATA_PATH = BASE_DIR / "data" / "processed" / "consolidated_dataset.csv"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    reg = joblib.load(MODELS_DIR / "regression_model.joblib")
    clf = joblib.load(MODELS_DIR / "classification_model.joblib")
    rec = joblib.load(MODELS_DIR / "recommender.joblib")
    return reg, clf, rec


df = load_data()
reg_model, clf_model, rec_artifacts = load_models()

st.title("🧳 Tourism Experience Analytics")
st.caption("Personalized attraction ratings, visit-mode prediction, and recommendations.")

tab_predict, tab_recommend, tab_eda = st.tabs(
    ["🔮 Predict", "✨ Recommendations", "📊 Explore the Data"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Predict rating & visit mode
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Tell us about the trip")

    col1, col2, col3 = st.columns(3)
    with col1:
        continent = st.selectbox("Continent", sorted(df["Continent"].dropna().unique()))
        region_opts = sorted(df.loc[df["Continent"] == continent, "Region"].dropna().unique())
        region = st.selectbox("Region", region_opts)
    with col2:
        country_opts = sorted(df.loc[df["Region"] == region, "Country"].dropna().unique())
        country = st.selectbox("Country", country_opts)
        attraction_type = st.selectbox("Attraction Type", sorted(df["AttractionType"].dropna().unique()))
    with col3:
        visit_year = st.number_input("Visit Year", min_value=2015, max_value=2030, value=2025)
        visit_month = st.selectbox("Visit Month", list(range(1, 13)), index=5)

    # Use dataset-level averages as reasonable defaults for engineered features
    attr_subset = df.loc[df["AttractionType"] == attraction_type]
    default_user_visits = int(df["UserVisitCount"].median()) if not df["UserVisitCount"].empty else 5
    default_attr_avg = float(attr_subset["AttractionAvgRating"].mean()) if not attr_subset.empty and not pd.isna(attr_subset["AttractionAvgRating"].mean()) else 3.5
    default_attr_pop = int(attr_subset["AttractionPopularity"].mean()) if not attr_subset.empty and not pd.isna(attr_subset["AttractionPopularity"].mean()) else 10

    input_row = pd.DataFrame([{
        "VisitYear": visit_year,
        "VisitMonth": visit_month,
        "UserVisitCount": default_user_visits,
        "AttractionAvgRating": default_attr_avg,
        "AttractionPopularity": default_attr_pop,
        "Continent": continent,
        "Region": region,
        "Country": country,
        "AttractionType": attraction_type,
    }])

    if st.button("Predict", type="primary"):
        pred_rating = reg_model.predict(input_row)[0]
        pred_mode = clf_model.predict(input_row)[0]
        mode_proba = clf_model.predict_proba(input_row)[0]
        mode_classes = clf_model.named_steps["model"].classes_

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Attraction Rating", f"{pred_rating:.2f} / 5")
        with c2:
            st.metric("Predicted Visit Mode", pred_mode)

        proba_df = pd.DataFrame({"VisitMode": mode_classes, "Probability": mode_proba}).sort_values(
            "Probability", ascending=False
        )
        st.bar_chart(proba_df.set_index("VisitMode"))

# ---------------------------------------------------------------------------
# TAB 2 — Recommendations
# ---------------------------------------------------------------------------
with tab_recommend:
    st.subheader("Get personalized attraction recommendations")
    known_users = sorted(rec_artifacts["user_item_matrix"].index.tolist())
    mode = st.radio("Choose input mode", ["Pick an existing UserId", "New / unknown user (cold start)"])

    if mode == "Pick an existing UserId":
        user_id = st.selectbox("UserId", known_users)
    else:
        user_id = -1  # guaranteed not in matrix -> triggers cold-start fallback

    top_n = st.slider("Number of recommendations", 3, 15, 5)

    if st.button("Recommend attractions"):
        recs = recommend_for_user(user_id, rec_artifacts, top_n=top_n)
        st.dataframe(
            recs.reset_index()[["AttractionId", "Attraction", "AttractionType",
                                 "AttractionCityName", "AttractionAvgRating", "AttractionPopularity"]],
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# TAB 3 — EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ratings distribution**")
        fig, ax = plt.subplots()
        df["Rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with c2:
        st.markdown("**Visit mode distribution**")
        fig, ax = plt.subplots()
        df["VisitMode"].value_counts().plot(kind="bar", ax=ax, color="orange")
        ax.set_xlabel("Visit Mode")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.markdown("**Top attraction types by average rating**")
    top_types = df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False)
    st.bar_chart(top_types)

    st.markdown("**Users by continent**")
    st.bar_chart(df.drop_duplicates("UserId")["Continent"].value_counts())

    with st.expander("Raw consolidated dataset (sample)"):
        st.dataframe(df.sample(min(200, len(df))), use_container_width=True)
