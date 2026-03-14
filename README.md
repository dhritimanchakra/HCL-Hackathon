#  House Price Prediction System

A full-stack machine learning project that predicts house sale prices using regression models trained on the Ames Housing dataset. The system includes data preprocessing, model training, a prediction API, and a web-based interface for user interaction.

---

#  Project Overview

This project builds a machine learning regression pipeline to estimate housing sale prices based on structural house features and amenities.

The workflow consists of:

1. Data preprocessing and feature engineering
2. Model training and evaluation
3. Model deployment via API
4. Interactive frontend for prediction

The system uses several regression models and selects the best-performing model for deployment.

---

#  Project Structure

```
House-Price-Predictor
│
├── preprocess.py          # Dataset cleaning & feature engineering
├── train_models.py        # Model training and evaluation
├── best_model.pkl         # Saved trained model
│
├── api_server.py          # FastAPI backend for predictions
├── index.html             # Web interface for prediction
│
├── processed_house_data.csv
├── final_dataset.csv
│
└── README.md
```

---

#  Dataset

The project uses the **Ames Housing Dataset**, which contains detailed information about residential homes including:

* Total House area
* Lot size
* Overall house quality
* Garage capacity
* Bathrooms
* Bedrooms
* Total rooms
* Amenities such as pool, garden, gym, fireplace

Target variable:

**SalePrice → Final selling price of the house**

---

#  Step 1 — Data Preprocessing

The preprocessing pipeline prepares the dataset for machine learning.

## Cleaning

* Removed duplicate records
* Filled missing numeric values using median
* Filled missing categorical values with `"Unknown"`

## Timeline Adjustment

The original dataset years ranged from **2006–2010**.

To simulate modern predictions:

```
Yr Sold + 15 years
```

This shifts the dataset to approximately **2021–2025**.

## Location Classification

Zoning information is mapped to a simplified location feature:

Urban zones:

* RH
* RM
* C
* I

Rural zones:

* RL
* A
* FV

This creates a new feature:

```
location → Urban / Rural
```

---

#  Feature Engineering

Additional derived features were created to improve predictive performance.

### House Age

```
house_age = sale_year - Year Built
```

### Future Age

```
house_age_future = prediction_year - Year Built
```

### Amenities

Binary features were created to represent amenities:

| Feature       | Logic                                  |
| ------------- | -------------------------------------- |
| has_pool      | Urban houses with large living area    |
| has_gym       | Urban houses with moderate living area |
| has_garden    | Large land area                        |
| has_garage    | Garage capacity > 0                    |
| has_fireplace | Fireplace count > 0                    |

---

#  Final Features Used for Training

```
living_area
land_area
quality_score
garage_capacity
garage_area
bathrooms
bedrooms
total_rooms
has_pool
has_gym
has_garden
has_garage
has_fireplace
sale_year
house_age_future
location
```

Target variable:

```
SalePrice
```

Outliers were removed for houses with extremely large living areas.

---

#  Step 2 — Model Training

The dataset was split into training and testing sets:

```
80% Training
20% Testing
```

Models used:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor

Each model was trained and evaluated using regression metrics.

---

#  Model Evaluation Metrics

Three metrics were used to evaluate performance:

### R² Score

Measures how well the model explains price variation.

### RMSE (Root Mean Squared Error)

Measures the average prediction error.

### MAE (Mean Absolute Error)

Average difference between predicted and actual price.

Accuracy percentage was derived from the R² score.

---

#  Best Model

The **CatBoost Regressor** achieved the best performance and was selected as the final model.

The trained model was saved using:

```
joblib.dump(best_model, "best_model.pkl")
```

---

#  Step 3 — Prediction API

A backend API was developed using **FastAPI**.

Endpoint:

```
POST /predict
```

The API receives house features and returns the predicted price.

Example request:

```
{
  "living_area": 1800,
  "land_area": 5000,
  "quality_score": 7,
  "garage_capacity": 2,
  "garage_area": 400,
  "bathrooms": 2,
  "bedrooms": 3,
  "total_rooms": 7,
  "has_pool": 0,
  "has_gym": 0,
  "has_garden": 1,
  "has_garage": 1,
  "has_fireplace": 0,
  "sale_year": 2023,
  "house_age_future": 5,
  "location": 1
}
```

Response:

```
{
  "predicted_price": 245000
}
```

---

#  Step 4 — Web Interface

A simple frontend was built using **HTML + JavaScript**.

Features:

* User-friendly input form
* Toggle options for house amenities
* Real-time price prediction
* Visual display of predicted sale price

The frontend sends a request to the FastAPI backend and displays the predicted value.

---

# Technologies Used

Python
Pandas
NumPy
Scikit-learn
XGBoost
CatBoost
FastAPI
Joblib
HTML
JavaScript
Matplotlib
Seaborn

---

#  Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Feature importance analysis
* Model ensemble techniques
* Deployment to cloud services
* Integration with real estate datasets

---

#  Conclusion

This project demonstrates a complete machine learning workflow from raw data preprocessing to deployment of a prediction API and interactive user interface. The trained regression model can estimate house prices with strong predictive performance.

The system showcases the integration of data science, machine learning, and web development to create a functional predictive analytics tool.

---

