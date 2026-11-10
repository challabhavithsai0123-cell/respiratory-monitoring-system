
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# 1. PATHS AND CONFIGURATION
# ==========================================================

DATA_FILE = Path("results/reports/respiratory_ml_dataset.csv")
MODEL_DIR = Path("results/models")
REPORT_DIR = Path("results/reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("INTELLIGENT RESPIRATORY MONITORING - MODEL TRAINING")
print("=" * 70)

print("Total samples:", len(df))
print("Unique recordings:", df["recording_id"].nunique())

required_columns = [
    "recording_id",
    "consensus_rr_bpm",
    "reference_rr_ann1_bpm",
    "reference_rr_ann2_bpm",
    "detected_rr_bpm"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================================
# 3. DEFINE FEATURES AND TARGET
# ==========================================================

excluded_columns = [
    "recording_id",
    "window_start_s",
    "window_end_s",
    "reference_rr_ann1_bpm",
    "reference_rr_ann2_bpm",
    "annotation_rr_difference_bpm",
    "annotation_disagreement_bpm",
    "consensus_rr_bpm",
    "label_quality"
]

feature_columns = [
    column
    for column in df.select_dtypes(include=[np.number]).columns
    if column not in excluded_columns
]

if not feature_columns:
    raise ValueError("No numerical features available.")

X = df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

if X.isnull().any().any():
    raise ValueError(
        "Feature dataset contains missing or infinite values."
    )

y = df["consensus_rr_bpm"]
groups = df["recording_id"]

print("\nFeatures used:", len(feature_columns))
print(feature_columns)


# ==========================================================
# 4. GROUP-BASED TRAIN / TEST SPLIT
# ==========================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()

train_data = df.iloc[train_indices].copy()
test_data = df.iloc[test_indices].copy()

train_groups = groups.iloc[train_indices]
test_groups = groups.iloc[test_indices]

train_recordings = set(train_groups)
test_recordings = set(test_groups)

assert train_recordings.isdisjoint(test_recordings)

if len(train_recordings) < N_SPLITS:
    raise ValueError(
        "Not enough training recordings for five-fold GroupKFold."
    )

print("\n" + "=" * 70)
print("HOLDOUT TRAINING / TEST SPLIT")
print("=" * 70)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training recordings:", len(train_recordings))
print("Testing recordings:", len(test_recordings))
print("Recording overlap:", len(train_recordings & test_recordings))


# ==========================================================
# 5. DEFINE REGRESSION MODELS
# ==========================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        loss="huber",
        random_state=RANDOM_STATE
    )
}


# ==========================================================
# 6. METRIC CALCULATION
# ==========================================================

def calculate_metrics(actual, predicted):

    return {
        "MAE": mean_absolute_error(actual, predicted),

        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),

        "R2": r2_score(actual, predicted)
    }


# ==========================================================
# 7. FIVE-FOLD GROUP CROSS-VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("FIVE-FOLD GROUP CROSS-VALIDATION")
print("=" * 70)

group_kfold = GroupKFold(n_splits=N_SPLITS)

cv_fold_results = []
oof_predictions = []

for fold, (cv_train_idx, cv_val_idx) in enumerate(
    group_kfold.split(
        X_train,
        y_train,
        groups=train_groups
    ),
    start=1
):

    X_cv_train = X_train.iloc[cv_train_idx]
    X_cv_val = X_train.iloc[cv_val_idx]

    y_cv_train = y_train.iloc[cv_train_idx]
    y_cv_val = y_train.iloc[cv_val_idx]

    cv_train_groups = set(
        train_groups.iloc[cv_train_idx]
    )

    cv_val_groups = set(
        train_groups.iloc[cv_val_idx]
    )

    assert cv_train_groups.isdisjoint(cv_val_groups)

    val_data = train_data.iloc[cv_val_idx].copy()

    print("\n" + "-" * 70)
    print("FOLD:", fold)
    print("Training samples:", len(cv_train_idx))
    print("Validation samples:", len(cv_val_idx))
    print("Training recordings:", len(cv_train_groups))
    print("Validation recordings:", len(cv_val_groups))

    # ------------------------------------------------------
    # Store validation sample information
    # ------------------------------------------------------

    fold_predictions = val_data[
        [
            "recording_id",
            "window_start_s",
            "consensus_rr_bpm",
            "reference_rr_ann1_bpm",
            "reference_rr_ann2_bpm"
        ]
    ].copy()

    fold_predictions.insert(0, "fold", fold)

    # ------------------------------------------------------
    # Signal-processing baseline
    # ------------------------------------------------------

    baseline_predictions = (
        val_data["detected_rr_bpm"].to_numpy()
    )

    fold_predictions[
        "signal_processing_baseline"
    ] = baseline_predictions

    baseline_metrics = calculate_metrics(
        y_cv_val,
        baseline_predictions
    )

    cv_fold_results.append({
        "fold": fold,
        "model": "Signal Processing Baseline",
        "MAE": baseline_metrics["MAE"],
        "RMSE": baseline_metrics["RMSE"],
        "R2": baseline_metrics["R2"]
    })

    # ------------------------------------------------------
    # Machine-learning models
    # ------------------------------------------------------

    for name, model in models.items():

        fold_model = clone(model)

        fold_model.fit(
            X_cv_train,
            y_cv_train
        )

        predicted = fold_model.predict(X_cv_val)

        prediction_column = (
            name.lower().replace(" ", "_")
        )

        fold_predictions[
            prediction_column
        ] = predicted

        metrics = calculate_metrics(
            y_cv_val,
            predicted
        )

        cv_fold_results.append({
            "fold": fold,
            "model": name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

        print(
            name,
            "| MAE:", round(metrics["MAE"], 4),
            "| RMSE:", round(metrics["RMSE"], 4),
            "| R2:", round(metrics["R2"], 4)
        )

    print(
        "Signal Processing Baseline",
        "| MAE:", round(baseline_metrics["MAE"], 4),
        "| RMSE:", round(baseline_metrics["RMSE"], 4),
        "| R2:", round(baseline_metrics["R2"], 4)
    )

    # Store predictions from this validation fold.
    oof_predictions.append(fold_predictions)


# ==========================================================
# 8. CROSS-VALIDATION FOLD SUMMARY
# ==========================================================

cv_results_df = pd.DataFrame(cv_fold_results)

cv_summary_df = (
    cv_results_df
    .groupby("model")
    .agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std")
    )
    .reset_index()
)

cv_results_df.to_csv(
    REPORT_DIR / "cross_validation_fold_results.csv",
    index=False
)

cv_summary_df.to_csv(
    REPORT_DIR / "cross_validation_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(cv_summary_df.to_string(index=False))


# ==========================================================
# 9. POOLED OUT-OF-FOLD PREDICTIONS
# ==========================================================

oof_df = pd.concat(
    oof_predictions,
    ignore_index=True
)

oof_df.to_csv(
    REPORT_DIR / "pooled_oof_predictions.csv",
    index=False
)

print("\n" + "=" * 70)
print("POOLED OUT-OF-FOLD EVALUATION")
print("=" * 70)

print(
    "Total pooled validation samples:",
    len(oof_df)
)

print(
    "Unique validation recordings:",
    oof_df["recording_id"].nunique()
)

# Every training sample must have exactly one OOF prediction.
if len(oof_df) != len(X_train):
    raise ValueError(
        "Pooled OOF sample count does not match training sample count."
    )

if oof_df.duplicated(
    subset=["recording_id", "window_start_s"]
).any():
    raise ValueError(
        "Duplicate samples detected in pooled OOF predictions."
    )


# ==========================================================
# 10. CALCULATE POOLED METRICS
# ==========================================================

prediction_columns = {
    "Signal Processing Baseline": "signal_processing_baseline",
    "Random Forest": "random_forest",
    "Extra Trees": "extra_trees",
    "Gradient Boosting": "gradient_boosting"
}

pooled_results = []

for name, column in prediction_columns.items():

    predicted = oof_df[column]
    actual = oof_df["consensus_rr_bpm"]

    metrics = calculate_metrics(
        actual,
        predicted
    )

    ann1_mae = mean_absolute_error(
        oof_df["reference_rr_ann1_bpm"],
        predicted
    )

    ann2_mae = mean_absolute_error(
        oof_df["reference_rr_ann2_bpm"],
        predicted
    )

    pooled_results.append({
        "model": name,
        "consensus_MAE": metrics["MAE"],
        "consensus_RMSE": metrics["RMSE"],
        "consensus_R2": metrics["R2"],
        "annotator_1_MAE": ann1_mae,
        "annotator_2_MAE": ann2_mae
    })

pooled_results_df = pd.DataFrame(
    pooled_results
).sort_values("consensus_MAE")

pooled_results_df.to_csv(
    REPORT_DIR / "pooled_oof_metrics.csv",
    index=False
)

print("\nPOOLED METRICS")

print(
    pooled_results_df.to_string(index=False)
)


# ==========================================================
# 11. SELECT BEST MACHINE-LEARNING MODEL
# ==========================================================

ml_pooled_results = pooled_results_df[
    pooled_results_df["model"] != "Signal Processing Baseline"
].copy()

best_name = ml_pooled_results.iloc[0]["model"]

best_ml_mae = ml_pooled_results.iloc[0]["consensus_MAE"]

baseline_mae = pooled_results_df.loc[
    pooled_results_df["model"] == "Signal Processing Baseline",
    "consensus_MAE"
].iloc[0]

print("\n" + "=" * 70)
print("MODEL SELECTION")
print("=" * 70)

print("Best machine-learning model:", best_name)
print("Best ML pooled MAE:", round(best_ml_mae, 4))
print("Baseline pooled MAE:", round(baseline_mae, 4))

if best_ml_mae < baseline_mae:

    print(
        "Result: Best ML model outperformed the baseline "
        "in pooled MAE."
    )

else:

    print(
        "Result: Signal-processing baseline outperformed "
        "the best ML model in pooled MAE."
    )


# ==========================================================
# 12. FINAL HOLDOUT TEST EVALUATION
# ==========================================================

print("\n" + "=" * 70)
print("FINAL HOLDOUT TEST EVALUATION")
print("=" * 70)

holdout_results = []
holdout_predictions = {}

for name, model in models.items():

    final_model = clone(model)

    final_model.fit(
        X_train,
        y_train
    )

    predicted = final_model.predict(X_test)

    metrics = calculate_metrics(
        y_test,
        predicted
    )

    ann1_metrics = calculate_metrics(
        test_data["reference_rr_ann1_bpm"],
        predicted
    )

    ann2_metrics = calculate_metrics(
        test_data["reference_rr_ann2_bpm"],
        predicted
    )

    holdout_predictions[name] = predicted

    holdout_results.append({
        "model": name,
        "consensus_MAE": metrics["MAE"],
        "consensus_RMSE": metrics["RMSE"],
        "consensus_R2": metrics["R2"],
        "annotator_1_MAE": ann1_metrics["MAE"],
        "annotator_2_MAE": ann2_metrics["MAE"]
    })

    print("\n", name)

    print(
        "Consensus MAE:",
        round(metrics["MAE"], 4)
    )

    print(
        "Consensus RMSE:",
        round(metrics["RMSE"], 4)
    )

    print(
        "Consensus R2:",
        round(metrics["R2"], 4)
    )

    print(
        "Annotator 1 MAE:",
        round(ann1_metrics["MAE"], 4)
    )

    print(
        "Annotator 2 MAE:",
        round(ann2_metrics["MAE"], 4)
    )


# ==========================================================
# 13. HOLDOUT SIGNAL-PROCESSING BASELINE
# ==========================================================

baseline = test_data[
    "detected_rr_bpm"
].to_numpy()

baseline_metrics = calculate_metrics(
    y_test,
    baseline
)

print("\nSIGNAL-PROCESSING BASELINE")

print(
    "Baseline MAE:",
    round(baseline_metrics["MAE"], 4)
)

print(
    "Baseline RMSE:",
    round(baseline_metrics["RMSE"], 4)
)

print(
    "Baseline R2:",
    round(baseline_metrics["R2"], 4)
)


# ==========================================================
# 14. SAVE HOLDOUT RESULTS
# ==========================================================

holdout_results_df = pd.DataFrame(
    holdout_results
)

holdout_results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

prediction_df = test_data[
    [
        "recording_id",
        "window_start_s",
        "consensus_rr_bpm",
        "reference_rr_ann1_bpm",
        "reference_rr_ann2_bpm"
    ]
].copy()

for name, predicted in holdout_predictions.items():

    prediction_df[
        name.lower().replace(" ", "_") + "_prediction"
    ] = predicted

prediction_df[
    "signal_processing_baseline"
] = baseline

prediction_df.to_csv(
    REPORT_DIR / "model_test_predictions.csv",
    index=False
)


# ==========================================================
# 15. TRAIN AND SAVE SELECTED MODEL
# ==========================================================

best_model = clone(
    models[best_name]
)

best_model.fit(
    X_train,
    y_train
)

joblib.dump(
    best_model,
    MODEL_DIR / "best_respiratory_model.joblib"
)


# ==========================================================
# 16. SAVE MODEL METADATA
# ==========================================================

metadata = {

    "selected_model": best_name,

    "selection_method": (
        "Lowest pooled MAE from five-fold GroupKFold "
        "cross-validation"
    ),

    "feature_columns": feature_columns,

    "random_state": RANDOM_STATE,

    "number_of_folds": N_SPLITS,

    "target": "consensus_rr_bpm",

    "target_description": (
        "Provisional mean of two respiratory "
        "annotation-derived rates"
    ),

    "training_recordings": sorted(
        train_recordings
    ),

    "testing_recordings": sorted(
        test_recordings
    ),

    "cross_validation_best_ml_mae": float(
        best_ml_mae
    ),

    "cross_validation_baseline_mae": float(
        baseline_mae
    )
}

with open(
    MODEL_DIR / "model_metadata.json",
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================================
# 17. FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print("\nCROSS-VALIDATION SUMMARY")

print(
    cv_summary_df.sort_values(
        "MAE_mean"
    ).to_string(index=False)
)

print("\nPOOLED OUT-OF-FOLD RESULTS")

print(
    pooled_results_df.to_string(index=False)
)

print("\nHOLDOUT TEST RESULTS")

print(
    holdout_results_df.sort_values(
        "consensus_MAE"
    ).to_string(index=False)
)

print("\nSelected machine-learning model:", best_name)

print("\nSaved files:")

saved_files = [
    "results/reports/cross_validation_fold_results.csv",
    "results/reports/cross_validation_summary.csv",
    "results/reports/pooled_oof_predictions.csv",
    "results/reports/pooled_oof_metrics.csv",
    "results/reports/model_comparison.csv",
    "results/reports/model_test_predictions.csv",
    "results/models/best_respiratory_model.joblib",
    "results/models/model_metadata.json"
]

for file_path in saved_files:
    print(file_path)

print("\nMODEL TRAINING AND VALIDATION COMPLETED")
