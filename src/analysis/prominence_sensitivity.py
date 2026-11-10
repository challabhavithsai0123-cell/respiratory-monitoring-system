
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = ROOT / "results" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 125
TOLERANCE = 0.75

PROMINENCES = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16]
RECORDINGS = ["05", "13", "27"]

sos = butter(
    4, [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)


def evaluate(reference, detected):
    reference = np.asarray(reference, dtype=int)
    detected = np.asarray(detected, dtype=int)

    nr = len(reference)
    nd = len(detected)

    if nr == 0 or nd == 0:
        return 0, 0.0, 0.0, 0.0

    distances = (
        np.abs(reference[:, None] - detected[None, :]) / FS
    )

    size = nr + nd
    cost = np.full((size, size), 1e6)

    valid = distances <= TOLERANCE
    cost[:nr, :nd] = np.where(valid, distances, 1e6)
    cost[:nr, nd:] = TOLERANCE / 2
    cost[nr:, :nd] = TOLERANCE / 2
    cost[nr:, nd:] = 0

    rows, cols = linear_sum_assignment(cost)

    matched = sum(
        1 for r, c in zip(rows, cols)
        if r < nr and c < nd and valid[r, c]
    )

    precision = matched / nd
    recall = matched / nr

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return matched, precision * 100, recall * 100, f1 * 100


results = []

for number in RECORDINGS:

    signal_file = next(
        DATA_DIR.rglob(f"bidmc_{number}_Signals.csv")
    )

    breath_file = signal_file.with_name(
        f"bidmc_{number}_Breaths.csv"
    )

    signals = pd.read_csv(signal_file)
    annotations = pd.read_csv(breath_file)

    raw = signals[" RESP"].to_numpy()
    filtered = sosfiltfilt(sos, raw)

    print(f"\nRECORDING {number}")
    print("-" * 75)

    for prominence in PROMINENCES:

        peaks, _ = find_peaks(
            filtered,
            distance=int(1.5 * FS),
            prominence=prominence
        )

        for annotator, column in [
            (1, "breaths ann1 [signal sample no]"),
            (2, " breaths ann2 [signal sample no]")
        ]:

            reference = (
                annotations[column]
                .dropna()
                .astype(int)
                .to_numpy()
            )

            matched, precision, recall, f1 = evaluate(
                reference, peaks
            )

            results.append({
                "recording": number,
                "annotator": annotator,
                "prominence": prominence,
                "detected_events": len(peaks),
                "reference_events": len(reference),
                "matched_events": matched,
                "precision_percent": precision,
                "recall_percent": recall,
                "f1_percent": f1
            })

            print(
                f"Prominence={prominence:.2f} | "
                f"Annotator={annotator} | "
                f"Detected={len(peaks)} | "
                f"Precision={precision:.2f}% | "
                f"Recall={recall:.2f}% | "
                f"F1={f1:.2f}%"
            )

df = pd.DataFrame(results)

output = OUTPUT_DIR / "prominence_sensitivity_results.csv"
df.to_csv(output, index=False)

print(f"\nResults saved to: {output}")
