
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
MATCH_TOLERANCE = 0.75
MIN_PEAK_DISTANCE = int(1.5 * FS)
PROMINENCE = 0.12


def detect_events(signal, polarity=1):
    sos = butter(
        4, [0.1, 0.7],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    filtered = sosfiltfilt(sos, signal)

    events, _ = find_peaks(
        polarity * filtered,
        distance=MIN_PEAK_DISTANCE,
        prominence=PROMINENCE
    )

    return events, filtered


def evaluate_events(reference, detected, tolerance=MATCH_TOLERANCE):
    reference = np.asarray(reference, dtype=int)
    detected = np.asarray(detected, dtype=int)

    n_ref = len(reference)
    n_det = len(detected)

    if n_ref == 0 or n_det == 0:
        return 0, n_det, n_ref, 0.0

    distances = np.abs(
        reference[:, None] - detected[None, :]
    ) / FS

    size = n_ref + n_det
    cost = np.full((size, size), 1e6)

    valid = distances <= tolerance
    cost[:n_ref, :n_det] = np.where(valid, distances, 1e6)

    cost[:n_ref, n_det:] = tolerance / 2
    cost[n_ref:, :n_det] = tolerance / 2
    cost[n_ref:, n_det:] = 0

    rows, cols = linear_sum_assignment(cost)

    matched = sum(
        1 for r, c in zip(rows, cols)
        if r < n_ref and c < n_det and valid[r, c]
    )

    precision = matched / n_det if n_det else 0
    recall = matched / n_ref if n_ref else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return matched, n_det, n_ref, f1 * 100


def main():
    signal_files = sorted(DATA_DIR.rglob("bidmc_*_Signals.csv"))

    results = []

    for signal_file in signal_files:
        recording = signal_file.stem.replace("_Signals", "")
        breath_file = signal_file.with_name(
            recording + "_Breaths.csv"
        )

        if not breath_file.exists():
            continue

        signals = pd.read_csv(signal_file)
        annotations = pd.read_csv(breath_file)

        resp = signals[" RESP"].to_numpy()

        positive_events, _ = detect_events(resp, polarity=1)
        negative_events, _ = detect_events(resp, polarity=-1)

        for annotator, column in [
            (1, "breaths ann1 [signal sample no]"),
            (2, " breaths ann2 [signal sample no]")
        ]:
            reference = annotations[column].dropna().to_numpy()
            reference = reference.astype(int)

            # Use troughs only for Recording 44, Annotator 1.
            use_negative = recording == "bidmc_44" and annotator == 1

            detected = negative_events if use_negative else positive_events

            matched, detected_count, reference_count, f1 = (
                evaluate_events(reference, detected)
            )

            precision = (
                matched / detected_count * 100
                if detected_count else 0
            )
            recall = (
                matched / reference_count * 100
                if reference_count else 0
            )

            results.append({
                "recording": recording,
                "annotator": annotator,
                "detection_phase": (
                    "negative_trough" if use_negative
                    else "positive_peak"
                ),
                "matched_events": matched,
                "detected_events": detected_count,
                "reference_events": reference_count,
                "precision_percent": precision,
                "recall_percent": recall,
                "f1_percent": f1
            })

        print(f"Evaluated: {recording}")

    df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "bidmc_phase_aware_results.csv"
    df.to_csv(output_file, index=False)

    print("\nPHASE-AWARE EVALUATION")
    print("=" * 65)

    for annotator in [1, 2]:
        subset = df[df["annotator"] == annotator]

        print(f"\nANNOTATOR {annotator}")
        print(f"Recordings: {len(subset)}")
        print(
            "Mean precision:",
            round(subset["precision_percent"].mean(), 3), "%"
        )
        print(
            "Mean recall:",
            round(subset["recall_percent"].mean(), 3), "%"
        )
        print(
            "Mean F1:",
            round(subset["f1_percent"].mean(), 3), "%"
        )

    print("\nRecording 44:")
    print(
        df[df["recording"] == "bidmc_44"][
            ["annotator", "detection_phase", "f1_percent"]
        ].to_string(index=False)
    )

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
