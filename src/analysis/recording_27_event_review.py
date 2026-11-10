
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"

signal_file = next(
    DATA_DIR.rglob("bidmc_27_Signals.csv")
)

breath_file = next(
    DATA_DIR.rglob("bidmc_27_Breaths.csv")
)

signals = pd.read_csv(signal_file)
annotations = pd.read_csv(breath_file)

FS = 125

START_TIME = 455.0
END_TIME = 480.0

ann1_column = "breaths ann1 [signal sample no]"
ann2_column = " breaths ann2 [signal sample no]"

ann1 = annotations[ann1_column].dropna().to_numpy(dtype=int)
ann2 = annotations[ann2_column].dropna().to_numpy(dtype=int)

ann1_times = ann1 / FS
ann2_times = ann2 / FS

ann1_region = ann1_times[
    (ann1_times >= START_TIME) & (ann1_times < END_TIME)
]

ann2_region = ann2_times[
    (ann2_times >= START_TIME) & (ann2_times < END_TIME)
]

print("=" * 70)
print("RECORDING 27 — ANNOTATION DISAGREEMENT REVIEW")
print("=" * 70)

print("\nANNOTATOR 1 EVENTS")
print(np.round(ann1_region, 3))

print("\nANNOTATOR 2 EVENTS")
print(np.round(ann2_region, 3))

print("\nEVENT COUNTS")
print("Annotator 1:", len(ann1_region))
print("Annotator 2:", len(ann2_region))

print("\nNEAREST EVENT COMPARISON")

for t1 in ann1_region:
    if len(ann2_region) == 0:
        break

    differences = np.abs(ann2_region - t1)
    nearest_index = np.argmin(differences)

    print(
        f"Ann1: {t1:.3f}s | "
        f"Nearest Ann2: {ann2_region[nearest_index]:.3f}s | "
        f"Difference: {differences[nearest_index]:.3f}s"
    )

print("\n" + "=" * 70)
print("REVIEW COMPLETED")
print("=" * 70)
