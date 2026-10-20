
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data" / "Freshly_cleaned.csv")
model = joblib.load(BASE_DIR / "models" / "flight_price_model.pkl")

FEATURES = [
    "airline", "from", "to", "class_category", "class",
    "day", "month", "route", "dep_hour", "arr_hour",
    "dep_period", "arr_period", "duration_in_min", "stops",
    "stops_category", "arr_daytime", "arr_daytime_category",
    "dep_daytime", "dep_daytime_category", "month_category"
]

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["price"], test_size=0.2, random_state=42
)

pred = model.predict(X_test)

results = df.loc[X_test.index, [
    "airline", "class_category", "stops_category"
]].copy()

results["actual"] = y_test
results["predicted"] = pred
results["absolute_error"] = np.abs(y_test.to_numpy() - pred)

print("\n--- Largest 10 prediction errors ---")
print(
    results.nlargest(10, "absolute_error")[
        ["airline", "class_category", "actual",
         "predicted", "absolute_error"]
    ].round(2).to_string(index=False)
)

print("\n--- Error by travel class ---")
print(
    results.groupby("class_category")
    .apply(lambda g: pd.Series({
        "samples": len(g),
        "MAE": mean_absolute_error(g["actual"], g["predicted"])
    }), include_groups=False)
    .round(2).to_string()
)

print("\n--- Error by airline ---")
print(
    results.groupby("airline")
    .apply(lambda g: pd.Series({
        "samples": len(g),
        "MAE": mean_absolute_error(g["actual"], g["predicted"])
    }), include_groups=False)
    .sort_values("MAE")
    .round(2).to_string()
)
print("\n--- Error by travel class and actual fare range ---")

results["fare_range"] = pd.cut(
    results["actual"],
    bins=[0, 5000, 10000, 20000, 40000, 60000, float("inf")],
    labels=[
        "0-5k",
        "5k-10k",
        "10k-20k",
        "20k-40k",
        "40k-60k",
        "60k+"
    ]
)

print(
    results.groupby(
        ["class_category", "fare_range"],
        observed=True
    ).agg(
        samples=("actual", "size"),
        MAE=("absolute_error", "mean")
    ).round(2).to_string()
)

from sklearn.metrics import mean_absolute_percentage_error, r2_score

print("\n--- Percentage error by travel class ---")

for class_name, group in results.groupby("class_category"):
    actual = group["actual"].to_numpy()
    predicted = group["predicted"].to_numpy()

    mape = mean_absolute_percentage_error(actual, predicted) * 100
    r2 = r2_score(actual, predicted)

    print(
        f"{class_name}: "
        f"MAPE = {mape:.2f}%, "
        f"R2 = {r2:.4f}, "
        f"Samples = {len(group)}"
    )

print("\n--- High-fare Business examples ---")

business_high = results[
    (results["class_category"] == "Business")
    & (results["actual"] >= 60000)
].copy()

business_high["percentage_error"] = (
    business_high["absolute_error"]
    / business_high["actual"]
) * 100

print(
    business_high.nlargest(10, "absolute_error")[
        ["airline", "actual", "predicted",
         "absolute_error", "percentage_error"]
    ].round(2).to_string(index=False)
)
