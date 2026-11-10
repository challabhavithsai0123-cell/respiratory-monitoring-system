
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "results" / "reports"

df = pd.read_csv(
    REPORT_DIR / "bidmc_evaluation_results.csv"
)

print("\nWEAKEST RECORDINGS — STANDARD EVALUATION")
print("=" * 85)

for annotator in [1, 2]:
    subset = df[df["annotator"] == annotator].copy()

    subset = subset.sort_values(
        "f1_percent",
        ascending=True
    ).head(10)

    print(f"\nANNOTATOR {annotator} — LOWEST 10 F1 SCORES")
    print("-" * 85)

    print(
        subset[
            [
                "recording",
                "detected_events",
                "reference_events",
                "matched_events",
                "precision_percent",
                "recall_percent",
                "f1_percent",
                "rate_error_bpm"
            ]
        ].to_string(index=False)
    )

# Save the combined report.
weak_records = []

for annotator in [1, 2]:
    subset = df[df["annotator"] == annotator].nsmallest(
        10, "f1_percent"
    )
    weak_records.append(subset)

weak_df = pd.concat(weak_records, ignore_index=True)

output = REPORT_DIR / "weak_recordings_analysis.csv"
weak_df.to_csv(output, index=False)

print(f"\nSaved report: {output}")
