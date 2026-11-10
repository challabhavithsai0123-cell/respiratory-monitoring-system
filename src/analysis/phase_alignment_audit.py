
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"

FS = 125
TOLERANCE = 0.75

sos = butter(
    4, [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)


def match_events(reference, detected):

    n, m = len(reference), len(detected)

    if n == 0 or m == 0:
        return 0, np.nan

    distances = np.abs(
        reference[:, None] - detected[None, :]
    )

    unmatched_cost = TOLERANCE + 1
    invalid_cost = 1e6

    cost = np.full((n + m, n + m), invalid_cost)

    cost[:n, :m] = np.where(
        distances <= TOLERANCE,
        distances,
        invalid_cost
    )

    cost[:n, m:] = unmatched_cost
    cost[n:, :m] = unmatched_cost
    cost[n:, m:] = 0

    rows, cols = linear_sum_assignment(cost)

    valid = (rows < n) & (cols < m)
    rows, cols = rows[valid], cols[valid]

    valid = distances[rows, cols] <= TOLERANCE
    errors = distances[rows[valid], cols[valid]]

    return len(errors), np.mean(errors) if len(errors) else np.nan


results = []

for signal_file in sorted(DATA_DIR.rglob("*_Signals.csv")):

    breath_file = signal_file.with_name(
        signal_file.name.replace("_Signals.csv", "_Breaths.csv")
    )

    if not breath_file.exists():
        continue

    signals = pd.read_csv(signal_file)
    annotations = pd.read_csv(breath_file)

    resp = signals[" RESP"].to_numpy()
    filtered = sosfiltfilt(sos, resp)

    positive, _ = find_peaks(
        filtered, distance=int(1.5 * FS), prominence=0.12
    )

    negative, _ = find_peaks(
        -filtered, distance=int(1.5 * FS), prominence=0.12
    )

    positive_times = positive / FS
    negative_times = negative / FS

    for j in range(2):

        reference = (
            annotations.iloc[:, j].dropna().to_numpy() / FS
        )

        p_matches, p_error = match_events(
            reference, positive_times
        )

        n_matches, n_error = match_events(
            reference, negative_times
        )

        count = len(reference)

        results.append({
            "recording": signal_file.stem,
            "annotator": j + 1,
            "positive_match_percent": 100 * p_matches / count,
            "negative_match_percent": 100 * n_matches / count,
            "positive_error_s": p_error,
            "negative_error_s": n_error
        })

df = pd.DataFrame(results)

output = ROOT / "results" / "reports" / "phase_alignment_audit.csv"
df.to_csv(output, index=False)

print("\nPHASE ALIGNMENT AUDIT")
print("=" * 90)

for annotator in [1, 2]:

    subset = df[df.annotator == annotator]

    print(f"\nANNOTATOR {annotator}")
    print("Mean positive alignment:",
          round(subset.positive_match_percent.mean(), 2), "%")
    print("Mean negative alignment:",
          round(subset.negative_match_percent.mean(), 2), "%")

print("\nRECORDINGS WHERE NEGATIVE ALIGNMENT IS BETTER")

for _, row in df.iterrows():
    if row.negative_match_percent > row.positive_match_percent:
        print(
            row.recording,
            "| Annotator", row.annotator,
            "| Positive:", round(row.positive_match_percent, 2),
            "% | Negative:", round(row.negative_match_percent, 2), "%"
        )

print("\nSaved:", output)
