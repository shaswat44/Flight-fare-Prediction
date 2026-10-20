
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "flight_price_model.pkl"
DATA_PATH = BASE_DIR / "data" / "Freshly_cleaned.csv"

FEATURES = [
    "airline", "from", "to", "class_category", "class",
    "day", "month", "route", "dep_hour", "arr_hour",
    "dep_period", "arr_period", "duration_in_min", "stops",
    "stops_category", "arr_daytime", "arr_daytime_category",
    "dep_daytime", "dep_daytime_category", "month_category"
]

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df["price"]

# Reproduce the same split only if the original training used these settings.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Loading trained model...")
model = joblib.load(MODEL_PATH)

print("Generating predictions...")
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===== MODEL QUALITY RESULTS =====")
print(f"Test rows : {len(X_test)}")
print(f"MAE       : Rs. {mae:,.2f}")
print(f"RMSE      : Rs. {rmse:,.2f}")
print(f"R2 score  : {r2:.4f}")
print("\nSample predictions:")
comparison = pd.DataFrame({
    "Actual fare": y_test.iloc[:10].to_numpy(),
    "Predicted fare": predictions[:10]
})
comparison["Absolute error"] = abs(
    comparison["Actual fare"] - comparison["Predicted fare"]
)
print(comparison.round(2).to_string(index=False))
