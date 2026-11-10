
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("results/reports/respiratory_features.csv")
OUTPUT_DIR = Path("results/reports")

df = pd.read_csv(INPUT_FILE)

# Preserve original labels.
df["reference_rr_ann1_bpm"] = pd.to_numeric(
    df["reference_rr_ann1_bpm"], errors="coerce"
)

df["reference_rr_ann2_bpm"] = pd.to_numeric(
    df["reference_rr_ann2_bpm"], errors="coerce"
)

# Create provisional consensus target.
df["consensus_rr_bpm"] = (
    df["reference_rr_ann1_bpm"] +
    df["reference_rr_ann2_bpm"]
) / 2

# Record annotation disagreement.
df["annotation_disagreement_bpm"] = (
    df["reference_rr_ann1_bpm"] -
    df["reference_rr_ann2_bpm"]
).abs()

# Mark questionable labels rather than silently deleting them.
df["label_quality"] = "accepted"

df.loc[
    (df["reference_rr_ann1_bpm"] <= 0) |
    (df["reference_rr_ann2_bpm"] <= 0),
    "label_quality"
] = "zero_label"

df.loc[
    df["annotation_disagreement_bpm"] > 2,
    "label_quality"
] = "high_disagreement"

# Exclude zero-label windows from initial supervised training.
training_df = df[
    (df["reference_rr_ann1_bpm"] > 0) &
    (df["reference_rr_ann2_bpm"] > 0)
].copy()

# Keep high-disagreement samples for now.
# Their influence will be checked during model evaluation.
output_file = OUTPUT_DIR / "respiratory_ml_dataset.csv"
quality_file = OUTPUT_DIR / "respiratory_label_quality_audit.csv"

training_df.to_csv(output_file, index=False)
df.to_csv(quality_file, index=False)

print("=" * 55)
print("MACHINE LEARNING DATASET PREPARATION")
print("=" * 55)

print("Original windows:", len(df))
print("Training windows:", len(training_df))
print("Excluded zero-label windows:", len(df) - len(training_df))

print("\nLabel quality:")
print(df["label_quality"].value_counts())

print("\nConsensus respiratory rate:")
print(training_df["consensus_rr_bpm"].describe())

print("\nUnique recordings:", training_df["recording_id"].nunique())

print("\nSaved training dataset:", output_file)
print("Saved quality audit:", quality_file)
