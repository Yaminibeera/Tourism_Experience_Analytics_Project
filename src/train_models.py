"""
train_models.py
----------------
Trains and evaluates:
  1. Regression model -> predicts Rating
  2. Classification model -> predicts VisitMode
  3. Recommendation system -> item-based collaborative filtering + content-based fallback

Saves fitted artifacts to models/ for use by the Streamlit app.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NUM_FEATURES = ["VisitYear", "VisitMonth", "UserVisitCount", "AttractionAvgRating", "AttractionPopularity"]
CAT_FEATURES = ["Continent", "Region", "Country", "AttractionType"]


def load_dataset():
    return pd.read_csv(PROCESSED_DIR / "consolidated_dataset.csv")


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


# ---------------------------------------------------------------------------
# 1. Regression: predict Rating
# ---------------------------------------------------------------------------
def train_regression(df: pd.DataFrame):
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df["Rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    best_name, best_pipe, best_r2 = None, None, -np.inf
    for name, model in candidates.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        results[name] = {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}
        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2

    joblib.dump(best_pipe, MODELS_DIR / "regression_model.joblib")
    print("Regression results:", json.dumps(results, indent=2))
    print(f"Best regression model: {best_name} (R2={best_r2:.4f})")
    return results, best_name


# ---------------------------------------------------------------------------
# 2. Classification: predict VisitMode
# ---------------------------------------------------------------------------
def train_classification(df: pd.DataFrame):
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df["VisitMode"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    }

    results = {}
    best_name, best_pipe, best_f1 = None, None, -np.inf
    for name, model in candidates.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        results[name] = {"Accuracy": round(acc, 4), "Precision": round(prec, 4),
                          "Recall": round(rec, 4), "F1": round(f1, 4)}
        if f1 > best_f1:
            best_name, best_pipe, best_f1 = name, pipe, f1

    joblib.dump(best_pipe, MODELS_DIR / "classification_model.joblib")
    print("Classification results:", json.dumps(results, indent=2))
    print(f"Best classification model: {best_name} (F1={best_f1:.4f})")
    return results, best_name


# ---------------------------------------------------------------------------
# 3. Recommendation: item-based collaborative filtering (user-item rating matrix)
#    with a content-based fallback for cold-start users/attractions.
# ---------------------------------------------------------------------------
def train_recommender(df: pd.DataFrame):
    ui = df.pivot_table(index="UserId", columns="AttractionId", values="Rating", aggfunc="mean")

    # Item-item cosine similarity computed on the (mean-centered) rating matrix
    filled = ui.fillna(0)
    centered = filled.sub(filled.mean(axis=1), axis=0)
    item_vectors = centered.T.values
    norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized = item_vectors / norms
    sim_matrix = normalized @ normalized.T
    item_ids = ui.columns.tolist()
    sim_df = pd.DataFrame(sim_matrix, index=item_ids, columns=item_ids)

    # Content profile per attraction (for cold start / content-based fallback)
    content = df.drop_duplicates("AttractionId").set_index("AttractionId")[
        ["Attraction", "AttractionType", "AttractionCityName", "AttractionAvgRating", "AttractionPopularity"]
    ]

    joblib.dump({"user_item_matrix": ui, "item_similarity": sim_df, "content": content},
                MODELS_DIR / "recommender.joblib")
    print(f"Recommender built: {ui.shape[0]} users x {ui.shape[1]} attractions, "
          f"similarity matrix {sim_df.shape}")


def recommend_for_user(user_id, artifacts, top_n=5):
    """Reference implementation used by the Streamlit app."""
    ui, sim_df, content = artifacts["user_item_matrix"], artifacts["item_similarity"], artifacts["content"]
    if user_id in ui.index:
        user_ratings = ui.loc[user_id].dropna()
        if len(user_ratings) > 0:
            scores = sim_df[user_ratings.index].dot(user_ratings) / sim_df[user_ratings.index].abs().sum(axis=1).replace(0, 1e-9)
            scores = scores.drop(index=user_ratings.index, errors="ignore")
            top = scores.sort_values(ascending=False).head(top_n)
            return content.loc[top.index]
    # Cold start / fallback: most popular highly-rated attractions
    fallback = content.sort_values(["AttractionAvgRating", "AttractionPopularity"], ascending=False).head(top_n)
    return fallback


if __name__ == "__main__":
    dataset = load_dataset()
    reg_results, reg_best = train_regression(dataset)
    clf_results, clf_best = train_classification(dataset)
    train_recommender(dataset)

    summary = {
        "regression": {"best_model": reg_best, "metrics": reg_results},
        "classification": {"best_model": clf_best, "metrics": clf_results},
    }
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved training_summary.json")
