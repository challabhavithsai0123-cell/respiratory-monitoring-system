
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

DATA_DIR = Path("data/raw/BIDMC/bidmc-1.0.0/bidmc_csv")
OUTPUT_DIR = Path("results/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

breaths = pd.read_csv(DATA_DIR / "bidmc_46_Breaths.csv")

ann1 = breaths.iloc[:, 0].dropna().to_numpy() / 125.0
ann2 = breaths.iloc[:, 1].dropna().to_numpy() / 125.0

tolerance = 0.75
n, m = len(ann1), len(ann2)

# One-to-one matching with explicit unmatched-event assignments
size = n + m
penalty = tolerance + 0.01
large_cost = 1_000_000

cost = np.full((size, size), large_cost)

distances = np.abs(ann1[:, None] - ann2[None, :])
cost[:n, :m] = np.where(
    distances <= tolerance,
    distances,
    large_cost
)

cost[:n, m:] = penalty
cost[n:, :m] = penalty
cost[n:, m:] = 0

rows, cols = linear_sum_assignment(cost)

matched_ann2 = set()

for r, c in zip(rows, cols):
    if r < n and c < m and distances[r, c] <= tolerance:
        matched_ann2.add(c)

unmatched_ann2 = [
    ann2[i] for i in range(m) if i not in matched_ann2
]

print("=" * 60)
print("RECORDING 46 - UNMATCHED ANNOTATOR 2 EVENTS")
print("=" * 60)

print("Annotator 1 total:", n)
print("Annotator 2 total:", m)
print("Matched events:", len(matched_ann2))
print("Unmatched Annotator 2:", len(unmatched_ann2))

print("\nUnmatched event times (seconds):")

for i, t in enumerate(unmatched_ann2, start=1):
    print(f"{i:02d}. {t:.3f}")

output = OUTPUT_DIR / "recording_46_unmatched_ann2.csv"

pd.DataFrame({
    "unmatched_annotator_2_time_s": unmatched_ann2
}).to_csv(output, index=False)

print("\nSaved:", output)
