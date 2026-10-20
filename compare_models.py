"""
Compare the saved Random Forest pipeline with Extra Trees and a tuned RF.

This script does not modify app.py or models/flight_price_model.pkl.

Split reconstruction
--------------------
The training notebook (notebook/01_data_understanding.ipynb) used:

    train_test_split(X, y, test_size=0.2, random_state=42)

with no stratify argument. test_error_analysis.py uses the same call on
the same FEATURES list. There is no stored test-index file, so this is a
reconstructed split from those settings, not a separately versioned holdout.

If the saved model's overall MAE on this split is close to the reported
₹1,434.21, that is supporting evidence that the reconstruction matches
the original evaluation. Otherwise treat the numbers as a new baseline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Freshly_cleaned.csv"
EXISTING_MODEL_PATH = BASE_DIR / "models" / "flight_price_model.pkl"
CANDIDATE_MODEL_PATH = BASE_DIR / "models" / "candidate_flight_price_model.pkl"

DROP_COLS = [
    "price",
    "Unnamed: 0",
    "airline_index",
    "route_index",
    "flight_no",
]

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

# Matches the fitted production pipeline (object vs numeric columns).
CATEGORICAL_COLS = [
    "airline",
    "from",
    "to",
    "class_category",
    "route",
    "dep_period",
    "arr_period",
    "stops_category",
    "arr_daytime_category",
    "dep_daytime_category",
    "month_category",
]

NUMERICAL_COLS = [
    "class",
    "day",
    "month",
    "dep_hour",
    "arr_hour",
    "duration_in_min",
    "stops",
    "arr_daytime",
    "dep_daytime",
]

FARE_BINS = [0, 5000, 10000, 20000, 40000, 60000, float("inf")]
FARE_LABELS = ["0-5k", "5k-10k", "10k-20k", "20k-40k", "40k-60k", "60k+"]

REPORTED_MAE = 1434.21
REPORTED_RMSE = 3127.39
REPORTED_R2 = 0.9811

# Original notebook Random Forest settings (production model).
EXISTING_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

EXTRA_TREES_PARAMS = {
    "n_estimators": 150,
    "max_depth": 24,
    "min_samples_leaf": 2,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

# Small, explicit search. Candidates are scored on a train-only validation split.
RF_TUNE_CANDIDATES = [
    {
        "n_estimators": 150,
        "max_depth": 24,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "max_features": 0.5,
        "random_state": 42,
        "n_jobs": -1,
    },
    {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    {
        "n_estimators": 150,
        "max_depth": 28,
        "min_samples_leaf": 2,
        "min_samples_split": 4,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    {
        "n_estimators": 200,
        "max_depth": 24,
        "min_samples_leaf": 4,
        "min_samples_split": 2,
        "max_features": 0.7,
        "random_state": 42,
        "n_jobs": -1,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the saved RF with Extra Trees and a tuned RF."
    )
    parser.add_argument(
        "--tune-sample-size",
        type=int,
        default=40000,
        help="Training subsample used while scoring RF candidates (default: 40000).",
    )
    parser.add_argument(
        "--save-candidate",
        action="store_true",
        help=(
            "If the best new model is clearly better, save it to "
            "models/candidate_flight_price_model.pkl. Never overwrites the "
            "production pickle."
        ),
    )
    return parser.parse_args()


def make_preprocessor():
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    return ColumnTransformer(
        [
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
            ("numerical", numerical_pipeline, NUMERICAL_COLS),
        ]
    )


def make_model(regressor):
    return Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("regressor", regressor),
        ]
    )


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def format_metrics(metrics):
    return (
        f"n={metrics['n']:,}  "
        f"MAE=₹{metrics['MAE']:,.2f}  "
        f"RMSE=₹{metrics['RMSE']:,.2f}  "
        f"R²={metrics['R2']:.4f}"
    )


def grouped_report(y_true, y_pred, groups, title):
    print(f"\n--- {title} ---")
    rows = []
    frame = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
            "group": pd.Series(groups).astype(str).to_numpy(),
        }
    )
    for name, part in frame.groupby("group", dropna=False):
        metrics = regression_metrics(part["actual"], part["predicted"])
        rows.append({"group": name, **metrics})
        print(f"{name:12} {format_metrics(metrics)}")
    return rows


def evaluate_predictions(name, y_true, y_pred, class_labels, train_seconds, infer_seconds):
    overall = regression_metrics(y_true, y_pred)
    print(f"\n========== {name} ==========")
    print(f"Training time:  {train_seconds}")
    print(f"Inference time: {infer_seconds:.2f} s")
    print(f"Overall:        {format_metrics(overall)}")

    class_rows = grouped_report(y_true, y_pred, class_labels, "By travel class")
    fare_range = pd.cut(y_true, bins=FARE_BINS, labels=FARE_LABELS)
    range_rows = grouped_report(y_true, y_pred, fare_range, "By actual fare range")

    print("\n--- By class and actual fare range ---")
    combo = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
            "class_category": pd.Series(class_labels).astype(str).to_numpy(),
            "fare_range": pd.cut(y_true, bins=FARE_BINS, labels=FARE_LABELS),
        }
    )
    for (class_name, bucket), part in combo.groupby(
        ["class_category", "fare_range"], observed=True
    ):
        metrics = regression_metrics(part["actual"], part["predicted"])
        print(f"{class_name:10} {str(bucket):8} {format_metrics(metrics)}")

    class_lookup = {row["group"]: row for row in class_rows}
    return {
        "name": name,
        "overall": overall,
        "by_class": class_lookup,
        "by_range": range_rows,
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "y_pred": np.asarray(y_pred, dtype=float),
    }


def fit_and_time(model, X_train, y_train):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return time.perf_counter() - start


def predict_and_time(model, X_test):
    start = time.perf_counter()
    pred = model.predict(X_test)
    return pred, time.perf_counter() - start


def reconstruct_split(df):
    missing = [col for col in FEATURES + ["price"] if col not in df.columns]
    if missing:
        raise SystemExit(f"Required columns are missing: {missing}")

    unused = [col for col in DROP_COLS if col in df.columns]
    leftover = [col for col in df.columns if col not in FEATURES + ["price"] + unused]
    if leftover:
        print(
            "Columns excluded from modeling (same policy as the notebook): "
            + ", ".join(unused + leftover)
        )

    X = df[FEATURES]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    class_test = df.loc[X_test.index, "class_category"]
    return X_train, X_test, y_train, y_test, class_test


def tune_random_forest(X_train, y_train, sample_size):
    """Select RF hyperparameters on a train-only validation split."""
    X_inner_train, X_val, y_inner_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
    )

    if sample_size and sample_size < len(X_inner_train):
        X_search, _, y_search, _ = train_test_split(
            X_inner_train,
            y_inner_train,
            train_size=sample_size,
            random_state=42,
        )
        search_note = (
            f"scoring {len(RF_TUNE_CANDIDATES)} candidates on "
            f"{len(X_search):,} inner-train rows / {len(X_val):,} val rows"
        )
    else:
        X_search, y_search = X_inner_train, y_inner_train
        search_note = (
            f"scoring {len(RF_TUNE_CANDIDATES)} candidates on "
            f"{len(X_search):,} inner-train rows / {len(X_val):,} val rows"
        )

    print(f"\nTuning Random Forest ({search_note})")
    ranked = []
    for i, params in enumerate(RF_TUNE_CANDIDATES, start=1):
        model = make_model(RandomForestRegressor(**params))
        fit_seconds = fit_and_time(model, X_search, y_search)
        pred, infer_seconds = predict_and_time(model, X_val)
        metrics = regression_metrics(y_val, pred)
        ranked.append((metrics["MAE"], params, metrics, fit_seconds, infer_seconds))
        print(
            f"  candidate {i}: {format_metrics(metrics)}  "
            f"fit={fit_seconds:.1f}s  infer={infer_seconds:.2f}s  "
            f"params={ {k: params[k] for k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features']} }"
        )

    ranked.sort(key=lambda item: item[0])
    best_mae, best_params, best_metrics, _, _ = ranked[0]
    print(
        f"Selected tuned RF params by validation MAE "
        f"(₹{best_mae:,.2f}): {best_params}"
    )
    return best_params, best_metrics


def candidate_is_justified(existing, challenger):
    """Require a real overall MAE gain without harming Business-class MAE."""
    existing_mae = existing["overall"]["MAE"]
    challenger_mae = challenger["overall"]["MAE"]
    if challenger_mae >= existing_mae * 0.99:
        return False, "overall MAE is not at least 1% better than the saved model"

    existing_biz = existing["by_class"].get("Business", {}).get("MAE")
    challenger_biz = challenger["by_class"].get("Business", {}).get("MAE")
    if existing_biz is not None and challenger_biz is not None:
        if challenger_biz > existing_biz * 1.01:
            return False, "Business-class MAE is worse than the saved model"
    return True, "overall MAE improved by >=1% without worsening Business MAE"


def print_limitations(split_matches_report):
    print("\n========== LIMITATIONS ==========")
    if split_matches_report:
        print(
            "The reconstructed split (test_size=0.2, random_state=42, no "
            "stratify) produces overall metrics close to the notebook report, "
            "so this comparison is treated as using the original evaluation "
            "split settings."
        )
    else:
        print(
            "LABEL: NEW REPRODUCIBLE BASELINE. The original split settings "
            "are known (test_size=0.2, random_state=42, no stratify), but "
            "this run does not reproduce the reported MAE/RMSE/R² closely "
            "enough to claim it is the original evaluation. Possible causes "
            "include sklearn version differences, a later CSV rewrite, or "
            "floating-point/environment changes. No saved test-index file "
            "exists."
        )
    print(
        "- test_model_quality.py is not present in this project; only "
        "test_error_analysis.py was available for split comparison."
    )
    print(
        "- RF tuning uses a small explicit grid, a train-only validation "
        "split, and (by default) a training subsample. It is not an "
        "exhaustive GridSearchCV on all 240k training rows."
    )
    print(
        "- Tree models can still overfit expensive Business fares; overall "
        "R² is dominated by the Economy/Business price gap."
    )
    print(
        "- Inference times include the full sklearn pipeline (impute + "
        "one-hot + regressor) on the reconstructed test set."
    )
    print(
        "- The production file models/flight_price_model.pkl is never "
        "overwritten by this script."
    )


def main():
    args = parse_args()
    print("Loading dataset and saved pipeline...")
    df = pd.read_csv(DATA_PATH)
    existing_model = joblib.load(EXISTING_MODEL_PATH)

    print(f"Rows: {len(df):,}  columns: {df.shape[1]}")
    print(f"Existing model type: {type(existing_model).__name__}")
    if hasattr(existing_model, "named_steps"):
        print(f"Existing pipeline steps: {list(existing_model.named_steps)}")
        regressor = existing_model.named_steps.get("regressor")
        if regressor is not None:
            print(
                "Existing regressor: "
                f"{type(regressor).__name__} "
                f"(n_estimators={getattr(regressor, 'n_estimators', None)}, "
                f"max_depth={getattr(regressor, 'max_depth', None)}, "
                f"min_samples_leaf={getattr(regressor, 'min_samples_leaf', None)})"
            )

    X_train, X_test, y_train, y_test, class_test = reconstruct_split(df)
    print(
        f"Reconstructed split: train={len(X_train):,}  test={len(X_test):,}  "
        "(test_size=0.2, random_state=42, no stratify)"
    )
    print(
        "Preprocessing is identical for new models: median impute on numeric "
        "columns, most-frequent impute + OneHotEncoder(handle_unknown='ignore') "
        "on categorical columns. Encoders/imputers are fit on training data only."
    )

    results = []

    existing_pred, existing_infer = predict_and_time(existing_model, X_test)
    existing_result = evaluate_predictions(
        name="Existing saved Random Forest",
        y_true=y_test,
        y_pred=existing_pred,
        class_labels=class_test,
        train_seconds="n/a (pretrained pickle; not retrained)",
        infer_seconds=existing_infer,
    )
    results.append(existing_result)

    mae_delta = abs(existing_result["overall"]["MAE"] - REPORTED_MAE)
    rmse_delta = abs(existing_result["overall"]["RMSE"] - REPORTED_RMSE)
    r2_delta = abs(existing_result["overall"]["R2"] - REPORTED_R2)
    split_matches_report = mae_delta < 5 and rmse_delta < 10 and r2_delta < 0.002
    print(
        "\nReported notebook metrics: "
        f"MAE=₹{REPORTED_MAE:.2f}, RMSE=₹{REPORTED_RMSE:.2f}, R²={REPORTED_R2:.4f}"
    )
    print(
        "Saved-model metrics on reconstructed split: "
        f"MAE=₹{existing_result['overall']['MAE']:.2f}, "
        f"RMSE=₹{existing_result['overall']['RMSE']:.2f}, "
        f"R²={existing_result['overall']['R2']:.4f}"
    )
    if split_matches_report:
        print("Split check: reconstructed-split metrics match the reported evaluation.")
    else:
        print(
            "Split check: metrics differ from the reported evaluation. "
            "Treat this run as a new reproducible baseline."
        )

    extra_model = make_model(ExtraTreesRegressor(**EXTRA_TREES_PARAMS))
    extra_fit = fit_and_time(extra_model, X_train, y_train)
    extra_pred, extra_infer = predict_and_time(extra_model, X_test)
    extra_result = evaluate_predictions(
        name="Extra Trees (new training)",
        y_true=y_test,
        y_pred=extra_pred,
        class_labels=class_test,
        train_seconds=f"{extra_fit:.1f} s",
        infer_seconds=extra_infer,
    )
    extra_result["model"] = extra_model
    extra_result["params"] = EXTRA_TREES_PARAMS
    results.append(extra_result)

    best_params, _ = tune_random_forest(
        X_train, y_train, sample_size=args.tune_sample_size
    )
    tuned_model = make_model(RandomForestRegressor(**best_params))
    tuned_fit = fit_and_time(tuned_model, X_train, y_train)
    tuned_pred, tuned_infer = predict_and_time(tuned_model, X_test)
    tuned_result = evaluate_predictions(
        name="Tuned Random Forest (new training)",
        y_true=y_test,
        y_pred=tuned_pred,
        class_labels=class_test,
        train_seconds=f"{tuned_fit:.1f} s",
        infer_seconds=tuned_infer,
    )
    tuned_result["model"] = tuned_model
    tuned_result["params"] = best_params
    results.append(tuned_result)

    print("\n========== SUMMARY (overall test MAE) ==========")
    for item in results:
        print(f"{item['name']:36} {format_metrics(item['overall'])}")

    new_models = [item for item in results if item["name"] != "Existing saved Random Forest"]
    best_new = min(new_models, key=lambda item: item["overall"]["MAE"])
    justified, reason = candidate_is_justified(existing_result, best_new)
    print(f"\nBest new model: {best_new['name']}")
    print(f"Promotion check: {reason}")

    if args.save_candidate and justified:
        CANDIDATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_new["model"], CANDIDATE_MODEL_PATH)
        print(
            f"Saved candidate pipeline to {CANDIDATE_MODEL_PATH} "
            "(production model left unchanged)."
        )
        print(
            "App compatibility: the candidate uses the same FEATURES list as "
            "app.py and the saved production pipeline."
        )
    elif args.save_candidate:
        print("No candidate was saved because the improvement is not justified.")
    else:
        print(
            "No candidate was saved. Re-run with --save-candidate if you want "
            "a justified challenger written to models/candidate_flight_price_model.pkl."
        )

    print_limitations(split_matches_report)


if __name__ == "__main__":
    main()
