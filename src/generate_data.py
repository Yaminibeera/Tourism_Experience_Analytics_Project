"""
generate_data.py
-----------------
Generates a realistic synthetic Tourism Dataset matching the schema described
in the project brief (Transaction, User, City, Type, VisitMode, Continent,
Country, Region, Item/Attraction tables).

Replace this with your real dataset by dropping CSVs with the same column
names into the data/raw/ folder and skipping this script.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Continent / Region / Country / City reference tables
# ---------------------------------------------------------------------------
continents = pd.DataFrame({
    "ContinentId": range(1, 6),
    "Continent": ["Asia", "Europe", "North America", "Africa", "Oceania"],
})

regions_raw = {
    1: ["South Asia", "Southeast Asia"],
    2: ["Western Europe", "Eastern Europe"],
    3: ["North America East", "North America West"],
    4: ["East Africa", "West Africa"],
    5: ["Australasia", "Pacific Islands"],
}
region_rows = []
rid = 1
for cid, names in regions_raw.items():
    for name in names:
        region_rows.append({"RegionId": rid, "Region": name, "ContinentId": cid})
        rid += 1
regions = pd.DataFrame(region_rows)

country_names = {
    "South Asia": ["India", "Sri Lanka", "Nepal"],
    "Southeast Asia": ["Thailand", "Vietnam", "Indonesia"],
    "Western Europe": ["France", "Germany", "Spain"],
    "Eastern Europe": ["Poland", "Romania"],
    "North America East": ["USA-East", "Canada-East"],
    "North America West": ["USA-West", "Canada-West"],
    "East Africa": ["Kenya", "Tanzania"],
    "West Africa": ["Nigeria", "Ghana"],
    "Australasia": ["Australia", "New Zealand"],
    "Pacific Islands": ["Fiji", "Samoa"],
}
country_rows = []
cid_counter = 1
for _, r in regions.iterrows():
    for cname in country_names[r["Region"]]:
        country_rows.append({"CountryId": cid_counter, "Country": cname, "RegionId": r["RegionId"]})
        cid_counter += 1
countries = pd.DataFrame(country_rows)

city_rows = []
city_id_counter = 1
for _, c in countries.iterrows():
    n_cities = RNG.integers(2, 4)
    for i in range(n_cities):
        city_rows.append({
            "CityId": city_id_counter,
            "CityName": f"{c['Country']} City {i+1}",
            "CountryId": c["CountryId"],
        })
        city_id_counter += 1
cities = pd.DataFrame(city_rows)

# ---------------------------------------------------------------------------
# 2. Attraction type & VisitMode reference tables
# ---------------------------------------------------------------------------
attraction_types = pd.DataFrame({
    "AttractionTypeId": range(1, 9),
    "AttractionType": ["Beach", "Museum", "Historical Site", "Park", "Religious Site",
                        "Adventure", "Shopping", "Wildlife"],
})

visit_modes = pd.DataFrame({
    "VisitModeId": range(1, 6),
    "VisitMode": ["Business", "Couples", "Family", "Friends", "Solo"],
})

# ---------------------------------------------------------------------------
# 3. Attractions (Item data)
# ---------------------------------------------------------------------------
n_attractions = 150
attractions = pd.DataFrame({
    "AttractionId": range(1, n_attractions + 1),
    "AttractionCityId": RNG.choice(cities["CityId"], n_attractions),
    "AttractionTypeId": RNG.choice(attraction_types["AttractionTypeId"], n_attractions),
})
attractions["Attraction"] = attractions.apply(
    lambda r: f"{attraction_types.set_index('AttractionTypeId').loc[r['AttractionTypeId'], 'AttractionType']} #{r['AttractionId']}",
    axis=1,
)
attractions["AttractionAddress"] = attractions["AttractionCityId"].map(
    cities.set_index("CityId")["CityName"]
) + " Main Road"

# ---------------------------------------------------------------------------
# 4. Users
# ---------------------------------------------------------------------------
n_users = 2000
users = pd.DataFrame({"UserId": range(1, n_users + 1), "CityId": RNG.choice(cities["CityId"], n_users)})
users = users.merge(cities, on="CityId").merge(countries, on="CountryId").merge(regions, on="RegionId")
users = users[["UserId", "ContinentId", "RegionId", "CountryId", "CityId"]]

# ---------------------------------------------------------------------------
# 5. Transactions (with some realistic signal baked in, plus noise/missingness)
# ---------------------------------------------------------------------------
n_transactions = 20000
tx_user = RNG.choice(users["UserId"], n_transactions)
tx_attraction = RNG.choice(attractions["AttractionId"], n_transactions)
tx_year = RNG.integers(2019, 2025, n_transactions)
tx_month = RNG.integers(1, 13, n_transactions)
tx_mode = RNG.choice(visit_modes["VisitModeId"], n_transactions, p=[0.15, 0.25, 0.3, 0.2, 0.1])

attr_type_lookup = attractions.set_index("AttractionId")["AttractionTypeId"]
tx_attr_type = pd.Series(tx_attraction).map(attr_type_lookup).values

# base rating depends on attraction type + visit mode with noise -> gives models real signal
type_bonus = {1: 0.6, 2: 0.1, 3: 0.3, 4: 0.2, 5: 0.0, 6: 0.4, 7: -0.2, 8: 0.5}
mode_bonus = {1: -0.2, 2: 0.4, 3: 0.3, 4: 0.2, 5: -0.1}
base = 3.2
ratings = (
    base
    + np.array([type_bonus[t] for t in tx_attr_type])
    + np.array([mode_bonus[m] for m in tx_mode])
    + RNG.normal(0, 0.6, n_transactions)
)
ratings = np.clip(np.round(ratings), 1, 5).astype(int)

transactions = pd.DataFrame({
    "TransactionId": range(1, n_transactions + 1),
    "UserId": tx_user,
    "VisitYear": tx_year,
    "VisitMonth": tx_month,
    "VisitMode": pd.Series(tx_mode).map(visit_modes.set_index("VisitModeId")["VisitMode"]).values,
    "AttractionId": tx_attraction,
    "Rating": ratings,
})

# introduce a bit of realistic messiness: missing values, duplicate-like noise, inconsistent casing
missing_idx = RNG.choice(transactions.index, size=int(0.02 * n_transactions), replace=False)
transactions.loc[missing_idx, "Rating"] = np.nan
case_idx = RNG.choice(transactions.index, size=int(0.05 * n_transactions), replace=False)
transactions.loc[case_idx, "VisitMode"] = transactions.loc[case_idx, "VisitMode"].str.upper()

# ---------------------------------------------------------------------------
# Save all raw tables
# ---------------------------------------------------------------------------
continents.to_csv(RAW_DIR / "continent.csv", index=False)
regions.to_csv(RAW_DIR / "region.csv", index=False)
countries.to_csv(RAW_DIR / "country.csv", index=False)
cities.to_csv(RAW_DIR / "city.csv", index=False)
attraction_types.to_csv(RAW_DIR / "type.csv", index=False)
visit_modes.to_csv(RAW_DIR / "visit_mode.csv", index=False)
attractions.to_csv(RAW_DIR / "item.csv", index=False)
users.to_csv(RAW_DIR / "user.csv", index=False)
transactions.to_csv(RAW_DIR / "transaction.csv", index=False)

print("Synthetic dataset generated in", RAW_DIR)
for f in RAW_DIR.glob("*.csv"):
    print(" -", f.name, pd.read_csv(f).shape)
