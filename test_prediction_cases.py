
import time
import joblib
import pandas as pd
from pathlib import Path

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

def period(hour):
    if hour < 7:
        return "Early_morning"
    if hour < 12:
        return "Morning"
    if hour < 17:
        return "Afternoon"
    return "Night"

print("Loading model and dataset...")
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Use actual dataset rows so categorical values match training data.
sample = df.sample(n=min(20, len(df)), random_state=42)

print(f"\nRunning tests on {len(sample)} real dataset examples...\n")

passed = 0
failed = 0
start = time.perf_counter()

for index, (_, row) in enumerate(sample.iterrows(), start=1):
    X = pd.DataFrame(
        [{feature: row[feature] for feature in FEATURES}],
        columns=FEATURES
    )

    try:
        prediction = float(model.predict(X)[0])
        actual = float(row["price"])

        assert prediction >= 0, "Negative predicted fare"
        assert pd.notna(prediction), "Prediction is NaN"
        assert prediction != float("inf"), "Prediction is infinite"

        passed += 1
        print(
            f"PASS {index:02d} | "
            f"Actual: Rs.{actual:.2f} | "
            f"Predicted: Rs.{prediction:.2f}"
        )

    except Exception as error:
        failed += 1
        print(f"FAIL {index:02d} | {type(error).__name__}: {error}")

elapsed = time.perf_counter() - start

print("\n" + "=" * 45)
print(f"Tests passed : {passed}")
print(f"Tests failed : {failed}")
print(f"Total time   : {elapsed:.2f} seconds")
print("=" * 45)

if failed:
    raise SystemExit("TEST RESULT: FAIL")
else:
    print("TEST RESULT: PASS")
