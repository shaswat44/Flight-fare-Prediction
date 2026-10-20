
from pathlib import Path
import pandas as pd
import joblib

# Project folder
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Freshly_cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "flight_price_model.pkl"

print("=" * 50)
print("FLIGHT PRICE PREDICTION - HEALTH CHECK")
print("=" * 50)

# Test 1: Check required files
assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"
assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

print("\nPASS: Dataset and model files exist.")

# Test 2: Load dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("Total missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

assert not df.empty, "Dataset is empty."
assert "price" in df.columns, (
    "Target column 'price' not found. Check column names."
)

print("PASS: Dataset loaded successfully.")

# Test 3: Load saved model
model = joblib.load(MODEL_PATH)

print("\nLoaded model type:", type(model).__name__)
print("PASS: Saved model loaded successfully.")

print("\n" + "=" * 50)
print("ALL HEALTH CHECKS COMPLETED")
print("=" * 50)
