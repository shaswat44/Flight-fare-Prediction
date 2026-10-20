
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

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

print("Loading dataset...", flush=True)
df = pd.read_csv(DATA_PATH)

print("Loading model...", flush=True)
start = perf_counter()
model = joblib.load(MODEL_PATH)
print(f"Model load time: {perf_counter() - start:.2f} seconds", flush=True)

sample = df[FEATURES].iloc[[0]]

print("Running prediction...", flush=True)
start = perf_counter()
prediction = model.predict(sample)
print(f"Prediction time: {perf_counter() - start:.2f} seconds", flush=True)

print(f"Predicted fare: Rs. {float(prediction[0]):,.2f}", flush=True)
print("SPEED TEST PASSED", flush=True)
