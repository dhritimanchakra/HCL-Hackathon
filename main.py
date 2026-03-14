from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("best_model.pkl")

class HouseFeatures(BaseModel):
    living_area: float
    land_area: float
    quality_score: float
    garage_capacity: float
    garage_area: float
    bathrooms: float
    bedrooms: float
    total_rooms: float
    has_pool: int
    has_gym: int
    has_garden: int
    has_garage: int
    has_fireplace: int
    sale_year: int
    house_age_future: float
    location: float

# @app.post("/predict")
# def predict(data: HouseFeatures):
#     features = np.array([[
#         data.living_area, data.land_area, data.quality_score,
#         data.garage_capacity, data.garage_area, data.bathrooms,
#         data.bedrooms, data.total_rooms, data.has_pool,
#         data.has_gym, data.has_garden, data.has_garage,
#         data.has_fireplace, data.sale_year, data.house_age_future,
#         data.location
#     ]])
#     prediction = model.predict(features)[0]
#     return {"predicted_price": round(float(prediction), 2)}


@app.post("/predict")
def predict(data: HouseFeatures):
    # 1. Convert Pydantic model to dict
    input_data = data.dict()
    
    # 2. Create DataFrame (This preserves column names)
    df = pd.DataFrame([input_data])
    
    # 3. IMPORTANT: Ensure columns are in the EXACT same order as X_train
    # Replace this list with the actual order of columns in your CSV (minus SalePrice)
    column_order = [
    "living_area",
    "land_area",
    "quality_score",
    "garage_capacity",
    "garage_area",
    "bathrooms",
    "bedrooms",
    "total_rooms",
    "has_pool",
    "has_gym",
    "has_garden",
    "has_garage",
    "has_fireplace",
    "sale_year",
    "house_age_future",
    "location"
]
    
    df = df[column_order]
    
    # 4. Predict
    prediction = model.predict(df)[0]
    
    return {"predicted_price": round(float(prediction), 2)}


app.mount("/", StaticFiles(directory=".", html=True), name="static")