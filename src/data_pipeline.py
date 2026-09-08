"""
data_pipeline.py
-----------------
Loads the raw tourism tables, cleans them, joins them into one consolidated
dataset, and engineers features used by the regression, classification, and
recommendation models.
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_raw():
    return {
        "continent": pd.read_csv(RAW_DIR / "continent.csv"),
        "region": pd.read_csv(RAW_DIR / "region.csv"),
        "country": pd.read_csv(RAW_DIR / "country.csv"),
        "city": pd.read_csv(RAW_DIR / "city.csv"),
        "type": pd.read_csv(RAW_DIR / "type.csv"),
        "visit_mode": pd.read_csv(RAW_DIR / "visit_mode.csv"),
        "item": pd.read_csv(RAW_DIR / "item.csv"),
        "user": pd.read_csv(RAW_DIR / "user.csv"),
        "transaction": pd.read_csv(RAW_DIR / "transaction.csv"),
    }


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    # Standardize categorical casing (e.g. "BUSINESS" -> "Business")
    tx["VisitMode"] = tx["VisitMode"].astype(str).str.strip().str.title()

    # Drop rows with no rating (target for regression) — can't train/evaluate on them
    before = len(tx)
    tx = tx.dropna(subset=["Rating"])
    dropped = before - len(tx)

    # Clip impossible ratings, ensure ints
    tx["Rating"] = tx["Rating"].clip(1, 5).round().astype(int)

    # Standardize year/month types
    tx["VisitYear"] = tx["VisitYear"].astype(int)
    tx["VisitMonth"] = tx["VisitMonth"].astype(int).clip(1, 12)

    # Drop exact duplicate transactions
    tx = tx.drop_duplicates(subset=["UserId", "AttractionId", "VisitYear", "VisitMonth"])

    print(f"clean_transactions: dropped {dropped} rows with missing rating; "
          f"{before - dropped - len(tx)} exact duplicates removed; {len(tx)} rows remain")
    return tx


def build_consolidated_dataset():
    raw = load_raw()
    tx = clean_transactions(raw["transaction"])

    # user table already carries ContinentId/RegionId/CountryId/CityId; only bring in
    # the *names* from each reference table to avoid duplicate-key merge collisions
    user_geo = (
        raw["user"]
        .merge(raw["city"][["CityId", "CityName"]], on="CityId", how="left")
        .merge(raw["country"][["CountryId", "Country"]], on="CountryId", how="left")
        .merge(raw["region"][["RegionId", "Region"]], on="RegionId", how="left")
        .merge(raw["continent"][["ContinentId", "Continent"]], on="ContinentId", how="left")
    )

    item_full = raw["item"].merge(raw["type"], on="AttractionTypeId", how="left")
    item_full = item_full.merge(
        raw["city"].rename(columns={"CityId": "AttractionCityId", "CityName": "AttractionCityName"}),
        on="AttractionCityId", how="left",
    )

    df = (
        tx.merge(user_geo, on="UserId", how="left")
        .merge(item_full, on="AttractionId", how="left")
    )

    # Feature engineering -----------------------------------------------------
    # User-level aggregate profile features (average rating per user, count of visits)
    user_stats = tx.groupby("UserId")["Rating"].agg(UserAvgRating="mean", UserVisitCount="count").reset_index()
    df = df.merge(user_stats, on="UserId", how="left")

    # Attraction-level popularity / average rating (previous average rating feature)
    attraction_stats = tx.groupby("AttractionId")["Rating"].agg(
        AttractionAvgRating="mean", AttractionPopularity="count"
    ).reset_index()
    df = df.merge(attraction_stats, on="AttractionId", how="left")

    df.to_csv(PROCESSED_DIR / "consolidated_dataset.csv", index=False)
    print(f"Consolidated dataset: {df.shape}. Saved to {PROCESSED_DIR / 'consolidated_dataset.csv'}")
    return df


if __name__ == "__main__":
    build_consolidated_dataset()
