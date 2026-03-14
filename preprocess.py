

# =========================================
# House Price Prediction - Final Preprocessing
# Ames Housing Dataset
# =========================================

import pandas as pd
import numpy as np

# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------

df = pd.read_csv("/kaggle/input/datasets/prevek18/ames-housing-dataset/AmesHousing.csv")

print("Original dataset shape:", df.shape)

# -----------------------------------------
# 2. Basic Cleaning
# -----------------------------------------

df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include=["object"]).columns
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# -----------------------------------------
# 3. Update Timeline (shift dataset to modern years)
# -----------------------------------------

# Ames data is 2006–2010, shift it to ~2021–2025
df["Yr Sold"] = df["Yr Sold"] + 15

# -----------------------------------------
# 4. Location Mapping
# -----------------------------------------

def map_location(zone):

    if zone in ["RH","RM","C","I"]:
        return "Urban"

    elif zone in ["RL","A","FV"]:
        return "Rural"

    else:
        return "Urban"

df["location"] = df["MS Zoning"].apply(map_location)

# -----------------------------------------
# 5. Time Feature
# -----------------------------------------

df["house_age"] = df["Yr Sold"] - df["Year Built"]

df["prediction_year"] = df["Yr Sold"]

df["house_age_future"] = df["prediction_year"] - df["Year Built"]

# -----------------------------------------
# 6. Derived Amenities
# -----------------------------------------

# Pool (urban houses with larger living area)
df["has_pool"] = ((df["location"] == "Urban") & (df["Gr Liv Area"] > 1000)).astype(int)

# Gym (urban houses)
df["has_gym"] = ((df["location"] == "Urban") & (df["Gr Liv Area"] > 750)).astype(int)

# Garden (large land area)
df["has_garden"] = (df["Lot Area"] > 8000).astype(int)

# Garage
df["has_garage"] = (df["Garage Cars"] > 0).astype(int)

# Fireplace
df["has_fireplace"] = (df["Fireplaces"] > 0).astype(int)

# -----------------------------------------
# 7. Select Important Features
# -----------------------------------------

df_model = df[[

    "Gr Liv Area",
    "Lot Area",
    "Overall Qual",
    "Garage Cars",
    "Garage Area",
    "Full Bath",
    "Bedroom AbvGr",
    "TotRms AbvGrd",

    "has_pool",
    "has_gym",
    "has_garden",
    "has_garage",
    "has_fireplace",

    "Yr Sold",
    "house_age_future",
    "location",

    "SalePrice"
]]

# -----------------------------------------
# 8. Rename Columns
# -----------------------------------------

df_model = df_model.rename(columns={

    "Gr Liv Area": "living_area",
    "Lot Area": "land_area",
    "Overall Qual": "quality_score",
    "Garage Cars": "garage_capacity",
    "Garage Area": "garage_area",
    "Full Bath": "bathrooms",
    "Bedroom AbvGr": "bedrooms",
    "TotRms AbvGrd": "total_rooms",
    "Yr Sold": "sale_year"

})

# -----------------------------------------
# 9. Remove Extreme Outliers
# -----------------------------------------

df_model = df_model[df_model["living_area"] < 4000]

# -----------------------------------------
# 10. Save Processed Dataset
# -----------------------------------------

df_model.to_csv("processed_house_data.csv", index=False)

print("Preprocessing complete")
print("Processed dataset shape:", df_model.shape)

print("\nFinal Features:")
print(df_model.columns)