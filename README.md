# 🧳 Tourism Experience Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

An end-to-end machine learning platform designed to analyze tourist behavior, predict attraction ratings, classify visitor travel personas, and deliver personalized destination recommendations through an interactive Streamlit web dashboard.

---

## 📌 Project Overview

In the travel and tourism industry, understanding traveler preferences and delivering tailored recommendations is key to enhancing visitor satisfaction and driving destination engagement. 

This project implements a complete, production-ready machine learning pipeline:
1. **Attraction Rating Prediction (Regression):** Predicts the numerical rating ($1.0 - 5.0$) a visitor is likely to assign to a specific attraction based on historical preferences, geographic features, attraction popularity, and seasonality.
2. **Travel Persona Classification (Multi-Class Classification):** Classifies the traveler's visit mode (`Business`, `Couples`, `Family`, `Friends`, `Solo`) to help tourism operators tailor marketing and package offerings.
3. **Personalized Attraction Recommender System:** Utilizes item-based collaborative filtering (cosine similarity over mean-centered user-item rating matrices) with a popularity/rating-weighted fallback for cold-start (new) users.
4. **Interactive Web Dashboard:** A multi-tab Streamlit application providing real-time inference, personalized recommendations, and comprehensive exploratory data analysis (EDA).

---

## 🏗️ Architecture & Workflow

```
┌─────────────────┐       ┌────────────────────────┐       ┌─────────────────────────┐
│   Raw Data /    │ ----> │  ETL & Feature         │ ----> │ Consolidated Dataset    │
│  Relational DB  │       │  Engineering Pipeline  │       │ (data/processed/*.csv)  │
└─────────────────┘       └────────────────────────┘       └───────────┬─────────────┘
                                                                       │
           ┌───────────────────────────────────────────────────────────┴─────────────────┐
           ▼                                           ▼                                 ▼
┌──────────────────────┐                   ┌───────────────────────┐         ┌───────────────────────┐
│  Regression Pipeline │                   │ Classification Engine │         │ Collaborative Filter  │
│  (Rating Predictor)  │                   │ (Visit Mode Persona)  │         │ (Recommender System)  │
└──────────┬───────────┘                   └───────────┬───────────┘         └───────────┬───────────┘
           │                                           │                                 │
           └───────────────────────────────────────────┼─────────────────────────────────┘
                                                       ▼
                                     ┌───────────────────────────────────┐
                                     │  Streamlit Web Application (UI)   │
                                     │  - Real-time Predictions          │
                                     │  - Dynamic Recommendations        │
                                     │  - Exploratory Visual Analytics   │
                                     └───────────────────────────────────┘
```

---

## 📊 Relational Data Schema

The platform processes relational data structured across normalized entities:

| Table | Description | Key Features |
|---|---|---|
| `transaction.csv` | Core interaction logs | `TransactionId`, `UserId`, `AttractionId`, `VisitYear`, `VisitMonth`, `VisitMode`, `Rating` |
| `user.csv` | User demographic profiles | `UserId`, `ContinentId`, `RegionId`, `CountryId`, `CityId` |
| `item.csv` | Attractions catalog | `AttractionId`, `Attraction`, `AttractionCityId`, `AttractionTypeId`, `AttractionAddress` |
| `type.csv` | Category reference | `AttractionTypeId`, `AttractionType` (Historical Site, Beach, Park, etc.) |
| `visit_mode.csv` | Travel mode reference | `VisitModeId`, `VisitMode` (`Business`, `Couples`, `Family`, `Friends`, `Solo`) |
| `city.csv` | City reference | `CityId`, `CityName`, `CountryId` |
| `country.csv` | Country reference | `CountryId`, `Country`, `RegionId` |
| `region.csv` | Region reference | `RegionId`, `Region`, `ContinentId` |
| `continent.csv` | Continent reference | `ContinentId`, `Continent` |

### Feature Engineering
During the ETL stage (`src/data_pipeline.py`), raw entities are joined and enriched with behavioral features:
- **`UserVisitCount`**: Total interaction frequency per user.
- **`UserAvgRating`**: Historical average rating given by the user.
- **`AttractionAvgRating`**: Overall average score achieved by each attraction.
- **`AttractionPopularity`**: Total visit count per attraction.
- **Categorical Encodings**: One-Hot Encoding across geographic hierarchies (`Continent`, `Region`, `Country`) and `AttractionType`.
- **Numerical Scaling**: Standard scaling applied to temporal and interaction counters.

---

## 🤖 Machine Learning Models & Evaluation

### 1. Rating Prediction (Regression)
Trained on 80/20 train-test splits using scikit-learn Pipelines (`StandardScaler` + `OneHotEncoder`):
- **Models Evaluated:** Linear Regression, Random Forest Regressor, Gradient Boosting Regressor.
- **Metrics Tracked:** $R^2$ score, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
- **Selected Model:** Stored in `models/regression_model.joblib`.

### 2. Travel Mode Persona (Classification)
Predicts whether a visit is `Business`, `Couples`, `Family`, `Friends`, or `Solo`:
- **Models Evaluated:** Logistic Regression, Random Forest Classifier.
- **Metrics Tracked:** Accuracy, Weighted Precision, Weighted Recall, Weighted F1 Score.
- **Selected Model:** Stored in `models/classification_model.joblib`.

### 3. Recommender Engine
- **Item-Based Collaborative Filtering:** Mean-centers the user-item rating matrix to account for rating bias, computes cosine similarity between attraction interaction vectors, and predicts top-$N$ attractions.
- **Cold-Start Fallback:** For new or unknown tourists, dynamically falls back to top-rated, high-popularity attractions.
- **Artifacts:** Stored in `models/recommender.joblib`.

---

## 📁 Repository Structure

```
Tourism_Experience_Analytics_Project/
├── app/
│   └── app.py                 # Streamlit web application
├── data/
│   ├── raw/                   # Raw relational tables (CSV format)
│   └── processed/             # Cleaned & joined consolidated dataset
├── models/
│   ├── regression_model.joblib        # Trained regression pipeline
│   ├── classification_model.joblib    # Trained classification pipeline
│   ├── recommender.joblib             # Recommendation matrix & content metadata
│   └── training_summary.json          # Benchmark evaluation metrics
├── src/
│   ├── generate_data.py       # Data simulation & benchmark generation engine
│   ├── data_pipeline.py       # Cleaning, joining, and feature engineering
│   └── train_models.py        # Model training, evaluation, and serialization
├── .gitignore                 # Environment and build artifact exclusions
├── README.md                  # Comprehensive project documentation
├── requirements.txt           # Python dependencies
├── run_app.bat                # Windows one-click app launcher
└── run_pipeline.bat           # Windows one-click pipeline runner
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone this repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/Yaminibeera/Tourism_Experience_Analytics_Project.git
cd Tourism_Experience_Analytics_Project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline

To rebuild the consolidated dataset and train the models from scratch:

```bash
# Run data cleaning & feature engineering
python src/data_pipeline.py

# Train models and save serialized artifacts
python src/train_models.py
```

*(Windows users can also simply double-click `run_pipeline.bat`)*

### 3. Launching the Web App

Start the interactive Streamlit application:

```bash
python -m streamlit run app/app.py
```

*(Windows users can also simply double-click `run_app.bat`)*

Access the application in your browser at: **`http://localhost:8501`**

---

## 🖥️ Web Application Features

1. **🔮 Predict Tab:**
   - Interactive selection of Continent, Region, Country, Attraction Type, and Travel Dates.
   - Outputs predicted rating with bounds check and predicted travel mode alongside class probability distributions.
2. **✨ Recommendations Tab:**
   - Select an existing user ID to generate personalized collaborative filtering recommendations.
   - Switch to new/cold-start mode to get ranked destination highlights.
   - Configurable recommendation count slider ($3$ to $15$).
3. **📊 Explore the Data (EDA) Tab:**
   - Rating frequency distribution and visit mode distributions.
   - Top attraction categories ranked by average rating.
   - Continental tourist origin analysis.
   - Expandable interactive raw data explorer.

---

## 🛠️ Tech Stack

- **Core:** Python 3.10+
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, Joblib
- **Visualization:** Matplotlib, Streamlit
- **Version Control:** Git, GitHub

---

## 👤 Author

**Yamini Beera**
- GitHub: [@Yaminibeera](https://github.com/Yaminibeera)
