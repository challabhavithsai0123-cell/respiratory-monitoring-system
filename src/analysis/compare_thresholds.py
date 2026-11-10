
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "results" / "reports"

df = pd.read_csv(
    REPORT_DIR / "full_dataset_sensitivity_details.csv"
)

comparison = df[
    df["prominence"].isin([0.08, 0.12])
].pivot(
    index=["recording", "annotator"],
    columns="prominence",
    values="f1_percent"
).reset_index()

comparison.columns.name = None

comparison = comparison.rename(columns={
    0.08: "f1_at_008",
    0.12: "f1_at_012"
})

comparison["f1_change"] = (
    comparison["f1_at_008"] - comparison["f1_at_012"]
)

print("\nRECORDING-LEVEL THRESHOLD COMPARISON")
print("=" * 75)

for annotator in [1, 2]:

    subset = comparison[
        comparison["annotator"] == annotator
    ]

    improved = (subset["f1_change"] > 0.000001).sum()
    worsened = (subset["f1_change"] < -0.000001).sum()
    unchanged = (subset["f1_change"].abs() <= 0.000001).sum()

    print(f"\nANNOTATOR {annotator}")
    print("Improved recordings:", improved)
    print("Worsened recordings:", worsened)
    print("Unchanged recordings:", unchanged)
    print(
        "Average F1 change:",
        round(subset["f1_change"].mean(), 3),
        "percentage points"
    )

    print("\nLargest improvements:")
    print(
        subset.nlargest(5, "f1_change")[
            ["recording", "f1_at_008", "f1_at_012", "f1_change"]
        ].to_string(index=False)
    )

    print("\nLargest reductions:")
    print(
        subset.nsmallest(5, "f1_change")[
            ["recording", "f1_at_008", "f1_at_012", "f1_change"]
        ].to_string(index=False)
    )

output = REPORT_DIR / "threshold_008_vs_012_comparison.csv"
comparison.to_csv(output, index=False)

print(f"\nSaved comparison: {output}")
