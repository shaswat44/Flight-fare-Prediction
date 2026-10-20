
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Freshly_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "flight_price_model.pkl"

# Features used by the trained model
FEATURES = [
    "airline",
    "from",
    "to",
    "class_category",
    "class",
    "day",
    "month",
    "route",
    "dep_hour",
    "arr_hour",
    "dep_period",
    "arr_period",
    "duration_in_min",
    "stops",
    "stops_category",
    "arr_daytime",
    "arr_daytime_category",
    "dep_daytime",
    "dep_daytime_category",
    "month_category",
]

print("=" * 55)
print("FLIGHT PRICE PREDICTION - MODEL TEST")
print("=" * 55)

# Load data and model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Check whether all expected input features exist
missing_features = [col for col in FEATURES if col not in df.columns]

if missing_features:
    raise ValueError(f"Missing model features: {missing_features}")

print("\nPASS: All 20 model input features are present.")

# Use an existing row only to smoke-test prediction
sample = df[FEATURES].iloc[[0]]
actual_price = float(df["price"].iloc[0])

prediction = model.predict(sample)
predicted_price = float(prediction[0])

print("\nSample flight inputs:")
print(sample.to_string(index=False))

print(f"\nActual dataset price:   Rs. {actual_price:,.2f}")
print(f"Predicted model price:  Rs. {predicted_price:,.2f}")

# Basic output checks
assert len(prediction) == 1, "Expected exactly one prediction."
assert pd.notna(predicted_price), "Prediction is not a valid number."
assert predicted_price >= 0, "Predicted fare is negative."

print("\nPASS: Model generated a valid, non-negative prediction.")
print("\nNOTE: One prediction checks functionality, not model accuracy.")
print("=" * 55)
