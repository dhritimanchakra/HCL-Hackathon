# House Price Prediction using Machine Learning

## Overview

This project focuses on predicting housing sale prices using machine learning regression techniques. The dataset was preprocessed, important features were engineered, and multiple regression algorithms were trained to determine the most accurate predictive model.

The goal of this project is to build a reliable system capable of estimating house prices using property features such as living area, land area, amenities, house age, and location.

The project was developed as part of a **Machine Learning Hackathon**.

---

# Dataset

The dataset used in this project is the **Ames Housing Dataset**, which contains detailed housing information such as structural characteristics, property features, and sale prices.

The dataset originally contains housing data sold between **2006 and 2010**.

To simulate modern housing conditions, the timeline was adjusted to represent **2021–2025 market conditions**.

Target Variable:

**SalePrice** – Final selling price of the house.

---

# Data Preprocessing

A complete preprocessing pipeline was implemented to prepare the dataset for machine learning.

## 1. Data Cleaning

* Removed duplicate rows
* Filled missing numeric values using the **median**
* Filled missing categorical values with **"Unknown"**

## 2. Timeline Adjustment

The original dataset contained house sales from **2006–2010**.

To simulate current housing market conditions:

```
Yr Sold = Yr Sold + 15
```

This shifts the dataset to approximately **2021–2025**.

---

## 3. Location Mapping

Zoning categories were converted into simplified **Urban / Rural classifications**.

Urban Zones:

* RH
* RM
* C
* I

Rural Zones:

* RL
* A
* FV

This created a new feature:

```
location → Urban / Rural
```

---

## 4. Time-Based Feature Engineering

New temporal features were created:

* **house_age** – Age of the house at sale time
* **prediction_year** – Year used for prediction
* **house_age_future** – Age of house relative to prediction year

These features help the model understand how property age influences price.

---

## 5. Derived Property Amenities

Additional amenity features were created:

| Feature       | Description                              |
| ------------- | ---------------------------------------- |
| has_pool      | Urban houses with large living area      |
| has_gym       | Urban houses with sufficient living area |
| has_garden    | Houses with large land area              |
| has_garage    | Houses with garage capacity              |
| has_fireplace | Houses with fireplaces                   |

These features simulate modern housing amenities that may influence pricing.

---

## 6. Feature Selection

Important features were selected to reduce noise and improve model performance.

Final features include:

* living_area
* land_area
* quality_score
* garage_capacity
* garage_area
* bathrooms
* bedrooms
* total_rooms
* has_pool
* has_gym
* has_garden
* has_garage
* has_fireplace
* sale_year
* house_age_future
* location

Target variable:

```
SalePrice
```

---

## 7. Outlier Removal

Extremely large houses were removed to prevent skewed predictions:

```
living_area < 4000
```

This helps stabilize model training.

---

# Machine Learning Models

Four regression algorithms were implemented:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor
4. Support Vector Regression (SVR)

These models were selected to compare:

* Simple statistical models
* Ensemble tree models
* Boosting-based models
* Kernel-based regression models

---

# Model Training Pipeline

The machine learning workflow follows these steps:

```
Dataset Loading
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Train-Test Split (80% / 20%)
      ↓
Model Training
      ↓
Prediction
      ↓
Model Evaluation
```

---

# Evaluation Metrics

The models were evaluated using several regression metrics:

### R² Score

Measures how well the model explains the variance in housing prices.

### MAE (Mean Absolute Error)

Average absolute difference between predicted and actual prices.

### RMSE (Root Mean Squared Error)

Measures prediction error while penalizing larger errors more heavily.

### MAPE (Mean Absolute Percentage Error)

Shows prediction error as a percentage.

### Explained Variance Score

Measures how much of the data variability is captured by the model.

---

# Model Performance Results

| Model             | MAE        | RMSE       | R² Score  |
| ----------------- | ---------- | ---------- | --------- |
| Linear Regression | 23,472     | 35,410     | 0.823     |
| Random Forest     | 17,673     | 27,374     | 0.894     |
| XGBoost           | **17,277** | **25,596** | **0.907** |
| SVR               | 58,169     | 87,277     | -0.071    |

Best Performing Model:

**XGBoost Regressor**

Performance:

* **R² Score:** 0.907
* **MAE:** 17,277
* **RMSE:** 25,596

This means the model explains **about 90% of the variance in housing prices**.

---

# Model Visualization

The project also includes several visualizations:

* Model error comparison graphs
* R² performance comparison
* Feature importance visualization
* Actual vs Predicted price plots
* Residual distribution analysis

These visualizations help understand model performance and prediction behavior.

---

# Model Export

The best performing model is saved using **Joblib**:

```
house_price_model.pkl
```

This allows the trained model to be reused for future predictions without retraining.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib

---

# Conclusion

This project demonstrates how machine learning regression models can be used to predict housing prices effectively. Through proper preprocessing, feature engineering, and model comparison, the system achieved a strong predictive performance with an **R² score above 0.90** using XGBoost.

The model can serve as a foundation for building real-world housing price prediction systems.

---

# Future Improvements

Possible improvements include:

* Advanced feature engineering
* Model stacking
* Incorporating economic indicators
* Deploying the model as a web application

