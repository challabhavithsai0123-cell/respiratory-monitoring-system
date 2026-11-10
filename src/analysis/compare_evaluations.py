
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "results" / "reports"

standard = pd.read_csv(
    REPORT_DIR / "bidmc_evaluation_results.csv"
)

phase = pd.read_csv(
    REPORT_DIR / "bidmc_phase_aware_results.csv"
)

# Normalize recording names before merging.
standard["recording"] = (
    standard["recording"]
    .str.replace("_Signals.csv", "", regex=False)
)

comparison = pd.merge(
    standard,
    phase,
    on=["recording", "annotator"],
    suffixes=("_standard", "_phase"),
    validate="one_to_one"
)

# Calculate the change in F1 score.
comparison["f1_improvement"] = (
    comparison["f1_percent_phase"]
    - comparison["f1_percent_standard"]
)

print("\nEVALUATION COMPARISON")
print("=" * 65)

print("Matched evaluation rows:", len(comparison))

for annotator in [1, 2]:
    subset = comparison[
        comparison["annotator"] == annotator
    ]

    standard_f1 = subset["f1_percent_standard"].mean()
    phase_f1 = subset["f1_percent_phase"].mean()
    improvement = phase_f1 - standard_f1

    print(f"\nANNOTATOR {annotator}")
    print(f"Standard mean F1: {standard_f1:.3f}%")
    print(f"Phase-aware mean F1: {phase_f1:.3f}%")
    print(f"Mean F1 difference: {improvement:+.3f}%")

# Show recordings whose F1 score changed.
changed = comparison[
    comparison["f1_improvement"].abs() > 0.000001
]

print("\nRECORDINGS WITH CHANGED F1")
print("=" * 65)

if changed.empty:
    print("No F1 changes found.")
else:
    print(
        changed[
            [
                "recording",
                "annotator",
                "detection_phase",
                "f1_percent_standard",
                "f1_percent_phase",
                "f1_improvement"
            ]
        ].to_string(index=False)
    )

output_file = REPORT_DIR / "evaluation_comparison.csv"
comparison.to_csv(output_file, index=False)

print(f"\nComparison saved to: {output_file}")
