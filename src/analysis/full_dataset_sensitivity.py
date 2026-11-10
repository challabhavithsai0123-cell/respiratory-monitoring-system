
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
MIN_DISTANCE = int(1.5 * FS)

PROMINENCES = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16]

ANNOTATORS = {
    1: "breaths ann1 [signal sample no]",
    2: " breaths ann2 [signal sample no]"
}

sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)


def evaluate_events(reference, detected):
    nr = len(reference)
    nd = len(detected)

    if nr == 0 or nd == 0:
        return 0

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

    return matched


signal_files = sorted(DATA_DIR.rglob("bidmc_*_Signals.csv"))

detailed_results = []

for signal_file in signal_files:

    recording = signal_file.stem.replace("_Signals", "")
    breath_file = signal_file.with_name(recording + "_Breaths.csv")

    if not breath_file.exists():
        continue

    signals = pd.read_csv(signal_file)
    annotations = pd.read_csv(breath_file)

    resp = signals[" RESP"].to_numpy()
    filtered = sosfiltfilt(sos, resp)

    references = {}

    for annotator, column in ANNOTATORS.items():
        references[annotator] = (
            annotations[column]
            .dropna()
            .astype(int)
            .to_numpy()
        )

    for prominence in PROMINENCES:

        detected, _ = find_peaks(
            filtered,
            distance=MIN_DISTANCE,
            prominence=prominence
        )

        for annotator in [1, 2]:

            reference = references[annotator]

            matched = evaluate_events(reference, detected)

            precision = matched / len(detected) if len(detected) else 0
            recall = matched / len(reference) if len(reference) else 0

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )

            detailed_results.append({
                "recording": recording,
                "annotator": annotator,
                "prominence": prominence,
                "detected_events": len(detected),
                "reference_events": len(reference),
                "matched_events": matched,
                "precision_percent": precision * 100,
                "recall_percent": recall * 100,
                "f1_percent": f1 * 100
            })

    print(f"Processed {recording}")

df = pd.DataFrame(detailed_results)

detail_file = OUTPUT_DIR / "full_dataset_sensitivity_details.csv"
df.to_csv(detail_file, index=False)

summary = (
    df.groupby(["annotator", "prominence"])
    .agg(
        mean_precision=("precision_percent", "mean"),
        mean_recall=("recall_percent", "mean"),
        mean_f1=("f1_percent", "mean"),
        total_detected=("detected_events", "sum"),
        total_reference=("reference_events", "sum"),
        total_matched=("matched_events", "sum")
    )
    .reset_index()
)

summary["pooled_precision"] = (
    summary["total_matched"] / summary["total_detected"] * 100
)

summary["pooled_recall"] = (
    summary["total_matched"] / summary["total_reference"] * 100
)

summary["pooled_f1"] = (
    2 * summary["pooled_precision"] * summary["pooled_recall"]
    / (summary["pooled_precision"] + summary["pooled_recall"])
)

summary_file = OUTPUT_DIR / "full_dataset_sensitivity_summary.csv"
summary.to_csv(summary_file, index=False)

print("\nFULL DATASET PROMINENCE SENSITIVITY")
print("=" * 90)

for annotator in [1, 2]:

    subset = summary[summary["annotator"] == annotator]

    print(f"\nANNOTATOR {annotator}")
    print(
        subset[
            [
                "prominence",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "pooled_f1"
            ]
        ].round(3).to_string(index=False)
    )

print("\nSaved detailed results:", detail_file)
print("Saved summary results:", summary_file)
