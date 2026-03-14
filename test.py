import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# -----------------------------------------
# 1. Load Dataset
# -----------------------------------------
data = pd.read_csv("C:\\Users\\Neo\\Saved Games\\models\\final_dataset.csv")
data["location"] = data["location"].map({
    "Urban": 1,
    "Rural": 0
})

X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# -----------------------------------------
# 2. Train Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# 3. Define Models
# -----------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

results = []

# -----------------------------------------
# 4. Train + Evaluate
# -----------------------------------------
for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    accuracy = r2 * 100

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Accuracy (%)": accuracy
    })

results_df = pd.DataFrame(results)

print(results_df)

# -----------------------------------------
# 5. Visualization
# -----------------------------------------
plt.figure(figsize=(14,6))

results_melt = results_df.melt(
    id_vars="Model",
    value_vars=["RMSE","MAE","R2","Accuracy (%)"],
    var_name="Metric",
    value_name="Score"
)
print(list(X_train.columns))

sns.barplot(data=results_melt, x="Model", y="Score", hue="Metric")

plt.title("Model Performance Comparison")
plt.show()


import joblib

# Save best model (CatBoost)
best_model = models["CatBoost"]
best_model.fit(X_train, y_train)
joblib.dump(best_model, "best_model.pkl")
print("Model saved!")
