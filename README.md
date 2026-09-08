# Tourism Experience Analytics

An end-to-end ML project that cleans tourism transaction data, engineers
features, and delivers three things through a Streamlit app:

1. **Regression** — predicts the rating a user would give an attraction.
2. **Classification** — predicts the visit mode (Business / Family / Couples / Friends / Solo).
3. **Recommendation** — item-based collaborative filtering (with a content-based
   cold-start fallback) that suggests attractions to a user.

## ⚠️ About the data

No dataset file was attached to the project brief, so this build ships with a
**synthetic dataset generator** (`src/generate_data.py`) that creates tables
matching the exact schema described in the brief (Transaction, User, City,
Type, VisitMode, Continent, Country, Region, Item). It bakes in real signal
(attraction type & visit mode influence rating) so every model has something
genuine to learn, but it is *not* real-world tourism data.

**To use your real dataset:** drop CSVs with the same column names into
`data/raw/` (see schema below) and skip `generate_data.py` — everything
downstream (`data_pipeline.py`, `train_models.py`, the app) reads from that
folder and will work unchanged as long as the column names match.

### Expected raw schema (`data/raw/*.csv`)
| File | Key columns |
|---|---|
| `transaction.csv` | TransactionId, UserId, VisitYear, VisitMonth, VisitMode, AttractionId, Rating |
| `user.csv` | UserId, ContinentId, RegionId, CountryId, CityId |
| `city.csv` | CityId, CityName, CountryId |
| `country.csv` | CountryId, Country, RegionId |
| `region.csv` | RegionId, Region, ContinentId |
| `continent.csv` | ContinentId, Continent |
| `type.csv` | AttractionTypeId, AttractionType |
| `visit_mode.csv` | VisitModeId, VisitMode |
| `item.csv` | AttractionId, AttractionCityId, AttractionTypeId, Attraction, AttractionAddress |

## Project structure
```
tourism_project/
├── data/
│   ├── raw/            # source CSVs (synthetic or your real data)
│   └── processed/      # cleaned & joined dataset (generated)
├── models/             # trained model artifacts (generated)
├── src/
│   ├── generate_data.py   # synthetic data generator (skip if you have real data)
│   ├── data_pipeline.py   # cleaning, joining, feature engineering
│   └── train_models.py    # trains regression, classification, recommender
├── app/
│   └── app.py             # Streamlit application
├── requirements.txt
└── README.md
```

## Setup & run

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset (skip this step if you added your own CSVs to data/raw/)
python src/generate_data.py

# 4. Clean data + engineer features
python src/data_pipeline.py

# 5. Train all three models
python src/train_models.py

# 6. Launch the app
python -m streamlit run app/app.py
# (Or on Windows, simply double-click run_app.bat)
```

> **Note for Windows Users:** If `streamlit` is not directly in your system PATH, running `python -m streamlit run app/app.py` ensures it runs properly with your active Python environment. Alternatively, double-click `run_app.bat` to launch the app directly, or `run_pipeline.bat` to re-run the data cleaning and model training pipeline.

The app opens at `http://localhost:8501` with three tabs:
- **Predict** — enter trip details, get a predicted rating + visit mode.
- **Recommendations** — pick an existing user (or simulate a new/cold-start
  user) and get ranked attraction suggestions.
- **Explore the Data** — ratings distribution, visit-mode breakdown, top
  attraction types, and user geography charts.

## Modeling notes

- **Regression**: Linear Regression, Random Forest, and Gradient Boosting are
  trained and compared on R², RMSE, and MAE; the best by R² is saved.
- **Classification**: Logistic Regression and Random Forest are compared on
  accuracy, precision, recall, and F1 (weighted); the best by F1 is saved.
- **Recommendation**: builds a user–item rating matrix, mean-centers it, and
  computes item–item cosine similarity for collaborative filtering. Falls
  back to the most popular, highest-rated attractions for cold-start users.
- Metrics from the synthetic data are modest by design (the dataset has
  realistic noise) — `models/training_summary.json` has the full comparison,
  and swapping in a real dataset should improve them noticeably.

## Extending this project
- Swap in the real Kaggle "Tourism Experience" dataset by matching the schema above.
- Add hyperparameter tuning (GridSearchCV/Optuna) for the regression/classification models.
- Add a hybrid recommender that blends collaborative + content-based scores with weights.
- Persist user interactions from the Streamlit app back into `data/raw/transaction.csv` to make the recommender improve over time.
