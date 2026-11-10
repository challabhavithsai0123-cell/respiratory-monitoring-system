
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
REPORT_DIR = ROOT / "results" / "reports"

FS = 125
TOLERANCE = 0.75
UNMATCHED_COST = TOLERANCE + 0.001
INVALID_COST = 1e6


def match_events(events_a, events_b):
    """One-to-one matching between two event-time arrays."""

    a = np.asarray(events_a, dtype=float)
    b = np.asarray(events_b, dtype=float)

    n = len(a)
    m = len(b)

    if n == 0 or m == 0:
        return 0, np.array([])

    # Augmented assignment matrix allows unmatched events.
    cost = np.full((n + m, m + n), INVALID_COST)

    differences = np.abs(a[:, None] - b[None, :])
    valid = differences <= TOLERANCE

    cost[:n, :m] = np.where(valid, differences, INVALID_COST)

    # Each reference event can remain unmatched.
    for i in range(n):
        cost[i, m + i] = UNMATCHED_COST

    # Each event in B can remain unmatched.
    for j in range(m):
        cost[n + j, j] = UNMATCHED_COST

    # Dummy-to-dummy assignments.
    cost[n:, m:] = 0

    rows, cols = linear_sum_assignment(cost)

    matched_differences = []

    for row, col in zip(rows, cols):
        if row < n and col < m and valid[row, col]:
            matched_differences.append(a[row] - b[col])

    return len(matched_differences), np.asarray(matched_differences)


def calculate_metrics(count_a, count_b, matched):
    precision = matched / count_a if count_a else 0
    recall = matched / count_b if count_b else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return precision * 100, recall * 100, f1 * 100


signal_files = sorted(DATA_DIR.rglob("bidmc_*_Signals.csv"))

results = []

print("=" * 75)
print("BIDMC ANNOTATION DISAGREEMENT AUDIT")
print("=" * 75)

for signal_file in signal_files:

    recording_number = int(signal_file.stem.split("_")[1])

    breath_file = signal_file.with_name(
        f"bidmc_{recording_number:02d}_Breaths.csv"
    )

    if not breath_file.exists():
        continue

    annotations = pd.read_csv(breath_file)

    ann1 = (
        annotations.iloc[:, 0]
        .dropna()
        .to_numpy(dtype=float) / FS
    )

    ann2 = (
        annotations.iloc[:, 1]
        .dropna()
        .to_numpy(dtype=float) / FS
    )

    matched, differences = match_events(ann1, ann2)

    precision, recall, f1 = calculate_metrics(
        len(ann1), len(ann2), matched
    )

    count_difference = len(ann2) - len(ann1)

    results.append({
        "recording": recording_number,
        "annotator_1_events": len(ann1),
        "annotator_2_events": len(ann2),
        "event_count_difference": count_difference,
        "matched_events": matched,
        "unmatched_annotator_1": len(ann1) - matched,
        "unmatched_annotator_2": len(ann2) - matched,
        "precision_percent": precision,
        "recall_percent": recall,
        "f1_percent": f1,
        "mean_timing_difference_s": (
            np.mean(np.abs(differences))
            if len(differences) else np.nan
        )
    })

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No recordings were successfully evaluated.")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

output_file = REPORT_DIR / "annotation_disagreement_audit.csv"
df.to_csv(output_file, index=False)

print("\nRECORDING-LEVEL RESULTS")
print("-" * 75)

print(
    df[
        [
            "recording",
            "annotator_1_events",
            "annotator_2_events",
            "matched_events",
            "f1_percent"
        ]
    ].round(3).to_string(index=False)
)

print("\nOVERALL SUMMARY")
print("-" * 75)

print(f"Recordings analyzed: {len(df)}")
print(f"Total Annotator 1 events: {df['annotator_1_events'].sum()}")
print(f"Total Annotator 2 events: {df['annotator_2_events'].sum()}")
print(f"Total matched events: {df['matched_events'].sum()}")

total_a = df["annotator_1_events"].sum()
total_b = df["annotator_2_events"].sum()
total_matched = df["matched_events"].sum()

p, r, f1 = calculate_metrics(total_a, total_b, total_matched)

print(f"Overall precision: {p:.3f}%")
print(f"Overall recall: {r:.3f}%")
print(f"Overall F1: {f1:.3f}%")

print("\nLARGEST ANNOTATION COUNT DIFFERENCES")
print("-" * 75)

print(
    df.assign(
        absolute_difference=df["event_count_difference"].abs()
    )
    .sort_values("absolute_difference", ascending=False)
    .head(10)[
        [
            "recording",
            "annotator_1_events",
            "annotator_2_events",
            "event_count_difference",
            "f1_percent"
        ]
    ]
    .to_string(index=False)
)

print("\nLOWEST ANNOTATOR AGREEMENT")
print("-" * 75)

print(
    df.nsmallest(10, "f1_percent")[
        [
            "recording",
            "annotator_1_events",
            "annotator_2_events",
            "matched_events",
            "f1_percent"
        ]
    ].round(3).to_string(index=False)
)

print("\n" + "=" * 75)
print("ANNOTATION DISAGREEMENT AUDIT COMPLETED")
print(f"Report saved: {output_file}")
print("=" * 75)
