
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

REPORTS = ROOT / "results" / "reports"
MODELS = ROOT / "results" / "models"
GRAPHS = ROOT / "results" / "graphs"

GRAPHS.mkdir(parents=True, exist_ok=True)

OOF_FILE = REPORTS / "pooled_oof_predictions.csv"
TEST_FILE = REPORTS / "model_test_predictions.csv"
DATASET_FILE = REPORTS / "respiratory_ml_dataset.csv"
MODEL_FILE = MODELS / "best_respiratory_model.joblib"
METADATA_FILE = MODELS / "model_metadata.json"

MODEL_COLUMNS = {
    "Signal Processing Baseline": "signal_processing_baseline",
    "Extra Trees": "extra_trees",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting"
}

TEST_MODEL_COLUMNS = {
    "Signal Processing Baseline": "signal_processing_baseline",
    "Extra Trees": "extra_trees_prediction",
    "Random Forest": "random_forest_prediction",
    "Gradient Boosting": "gradient_boosting_prediction"
}

TARGET = "consensus_rr_bpm"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("RESPIRATORY MONITORING - MODEL ERROR ANALYSIS")
print("=" * 65)

oof = pd.read_csv(OOF_FILE)
test = pd.read_csv(TEST_FILE)
dataset = pd.read_csv(DATASET_FILE)

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

model = joblib.load(MODEL_FILE)

print("\nLoaded files successfully.")
print("OOF samples:", len(oof))
print("Holdout samples:", len(test))
print("Dataset samples:", len(dataset))


# ============================================================
# VALIDATE DATA
# ============================================================

required = [
    "recording_id",
    "window_start_s",
    TARGET,
    "reference_rr_ann1_bpm",
    "reference_rr_ann2_bpm"
]

for name, frame in [("OOF", oof), ("TEST", test)]:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} file missing columns: {missing}")

if oof.duplicated(["recording_id", "window_start_s"]).any():
    raise ValueError("Duplicate OOF recording/window entries detected.")

if test.duplicated(["recording_id", "window_start_s"]).any():
    raise ValueError("Duplicate test recording/window entries detected.")


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "samples": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan
        }

    return {
        "samples": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted) if len(actual) > 1 else np.nan
    }


def evaluate_recordings(frame, columns, evaluation_name):
    rows = []

    for recording_id, group in frame.groupby("recording_id"):
        for model_name, prediction_column in columns.items():

            metrics = calculate_metrics(
                group[TARGET],
                group[prediction_column]
            )

            rows.append({
                "evaluation": evaluation_name,
                "recording_id": recording_id,
                "model": model_name,
                **metrics
            })

    return pd.DataFrame(rows)


# ============================================================
# RECORDING-WISE PERFORMANCE
# ============================================================

oof_recording_metrics = evaluate_recordings(
    oof, MODEL_COLUMNS, "OOF"
)

test_recording_metrics = evaluate_recordings(
    test, TEST_MODEL_COLUMNS, "Holdout"
)

recording_metrics = pd.concat(
    [oof_recording_metrics, test_recording_metrics],
    ignore_index=True
)

recording_metrics.to_csv(
    REPORTS / "recording_wise_error_analysis.csv",
    index=False
)

print("\nRecording-wise metrics saved.")


# ============================================================
# OVERALL ERROR SUMMARY
# ============================================================

summary_rows = []

for evaluation_name, frame, columns in [
    ("OOF", oof, MODEL_COLUMNS),
    ("Holdout", test, TEST_MODEL_COLUMNS)
]:

    for model_name, prediction_column in columns.items():

        metrics = calculate_metrics(
            frame[TARGET],
            frame[prediction_column]
        )

        summary_rows.append({
            "evaluation": evaluation_name,
            "model": model_name,
            **metrics
        })

overall_summary = pd.DataFrame(summary_rows)

overall_summary.to_csv(
    REPORTS / "error_analysis_summary.csv",
    index=False
)

print("\nOVERALL ERROR SUMMARY")
print(overall_summary.to_string(index=False))


# ============================================================
# MERGE ANNOTATION QUALITY INFORMATION
# ============================================================

quality_columns = [
    "recording_id",
    "window_start_s",
    "annotation_disagreement_bpm",
    "label_quality"
]

quality = dataset[quality_columns].copy()

for evaluation_name, frame, columns in [
    ("OOF", oof, MODEL_COLUMNS),
    ("Holdout", test, TEST_MODEL_COLUMNS)
]:

    merged = frame.merge(
        quality,
        on=["recording_id", "window_start_s"],
        how="left",
        validate="one_to_one"
    )

    quality_rows = []

    for quality_label, group in merged.groupby("label_quality"):

        for model_name, prediction_column in columns.items():

            metrics = calculate_metrics(
                group[TARGET],
                group[prediction_column]
            )

            quality_rows.append({
                "evaluation": evaluation_name,
                "label_quality": quality_label,
                "model": model_name,
                **metrics
            })

    pd.DataFrame(quality_rows).to_csv(
        REPORTS / f"annotation_quality_{evaluation_name.lower()}.csv",
        index=False
    )

print("\nAnnotation-quality analysis saved.")


# ============================================================
# ACTUAL VS PREDICTED PLOTS
# ============================================================

for evaluation_name, frame, columns in [
    ("OOF", oof, MODEL_COLUMNS),
    ("Holdout", test, TEST_MODEL_COLUMNS)
]:

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    minimum = min(
        frame[TARGET].min(),
        min(frame[c].min() for c in columns.values())
    )

    maximum = max(
        frame[TARGET].max(),
        max(frame[c].max() for c in columns.values())
    )

    for ax, (model_name, prediction_column) in zip(
        axes, columns.items()
    ):

        ax.scatter(
            frame[TARGET],
            frame[prediction_column],
            alpha=0.45,
            s=14
        )

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            "k--",
            label="Ideal prediction"
        )

        ax.set_title(model_name)
        ax.set_xlabel("Reference respiratory rate (BPM)")
        ax.set_ylabel("Predicted respiratory rate (BPM)")
        ax.legend()
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{evaluation_name}: Actual vs Predicted Respiratory Rate"
    )

    fig.tight_layout()

    fig.savefig(
        GRAPHS / f"actual_vs_predicted_{evaluation_name.lower()}.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# RESIDUAL DISTRIBUTION
# ============================================================

for evaluation_name, frame, columns in [
    ("OOF", oof, MODEL_COLUMNS),
    ("Holdout", test, TEST_MODEL_COLUMNS)
]:

    fig, ax = plt.subplots(figsize=(11, 6))

    for model_name, prediction_column in columns.items():

        residuals = (
            frame[prediction_column] - frame[TARGET]
        )

        ax.hist(
            residuals,
            bins=35,
            alpha=0.35,
            label=model_name
        )

    ax.axvline(0, color="black", linestyle="--")
    ax.set_title(f"{evaluation_name}: Prediction Residual Distribution")
    ax.set_xlabel("Prediction error (BPM)")
    ax.set_ylabel("Number of windows")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        GRAPHS / f"residual_distribution_{evaluation_name.lower()}.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# RECORDING-WISE MAE COMPARISON
# ============================================================

for evaluation_name in ["OOF", "Holdout"]:

    subset = recording_metrics[
        recording_metrics["evaluation"] == evaluation_name
    ]

    pivot = subset.pivot(
        index="recording_id",
        columns="model",
        values="MAE"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(15, 7),
        width=0.8
    )

    ax.set_title(
        f"{evaluation_name}: Recording-wise MAE Comparison"
    )
    ax.set_xlabel("Recording")
    ax.set_ylabel("MAE (BPM)")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        GRAPHS / f"recording_wise_mae_{evaluation_name.lower()}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# EXTRA TREES FEATURE IMPORTANCE
# ============================================================

feature_columns = metadata["feature_columns"]

if not hasattr(model, "feature_importances_"):
    raise AttributeError(
        "Loaded Extra Trees model does not expose feature_importances_."
    )

if len(model.feature_importances_) != len(feature_columns):
    raise ValueError("Feature importance count does not match metadata.")

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(
    REPORTS / "extra_trees_feature_importance.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    importance["feature"][::-1],
    importance["importance"][::-1]
)

ax.set_title("Extra Trees Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()

fig.savefig(
    GRAPHS / "extra_trees_feature_importance.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("MODEL ERROR ANALYSIS COMPLETED")
print("=" * 65)

print("\nTOP FIVE EXTRA TREES FEATURES")
print(importance.head(5).to_string(index=False))

print("\nGENERATED REPORTS:")
print("- recording_wise_error_analysis.csv")
print("- error_analysis_summary.csv")
print("- annotation_quality_oof.csv")
print("- annotation_quality_holdout.csv")
print("- extra_trees_feature_importance.csv")

print("\nGENERATED GRAPHS:")
for file in sorted(GRAPHS.glob("*.png")):
    print("-", file.name)

print("\nAll analysis files saved successfully.")
