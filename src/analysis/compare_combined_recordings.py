
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "results" / "reports"

df = pd.read_csv(
    REPORT_DIR / "combined_detector_details.csv"
)

comparison = df[
    df["configuration"].isin(["A_original", "B_height_002"])
].pivot(
    index=["recording", "annotator"],
    columns="configuration",
    values="f1_percent"
).reset_index()

comparison.columns.name = None

comparison["f1_change"] = (
    comparison["B_height_002"] -
    comparison["A_original"]
)

print("\nCONFIGURATION B VS A — RECORDING-LEVEL ANALYSIS")
print("=" * 80)

for annotator in [1, 2]:

    subset = comparison[
        comparison["annotator"] == annotator
    ]

    improved = (subset["f1_change"] > 0.000001).sum()
    worsened = (subset["f1_change"] < -0.000001).sum()
    unchanged = (subset["f1_change"].abs() <= 0.000001).sum()

    print(f"\nANNOTATOR {annotator}")
    print("Improved:", improved)
    print("Worsened:", worsened)
    print("Unchanged:", unchanged)

    print(
        "Average F1 change:",
        round(subset["f1_change"].mean(), 3),
        "percentage points"
    )

    print("\nFive largest improvements:")
    print(
        subset.nlargest(5, "f1_change")[
            ["recording", "A_original", "B_height_002", "f1_change"]
        ].round(3).to_string(index=False)
    )

    print("\nFive largest reductions:")
    print(
        subset.nsmallest(5, "f1_change")[
            ["recording", "A_original", "B_height_002", "f1_change"]
        ].round(3).to_string(index=False)
    )

    print("\nFive weakest recordings with configuration B:")
    print(
        subset.nsmallest(5, "B_height_002")[
            ["recording", "A_original", "B_height_002", "f1_change"]
        ].round(3).to_string(index=False)
    )

output = REPORT_DIR / "combined_recording_comparison.csv"
comparison.to_csv(output, index=False)

print("\nSaved:", output)
