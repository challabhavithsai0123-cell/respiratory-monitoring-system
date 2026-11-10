
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
REPORT_DIR = ROOT / "results" / "reports"

FS = 125
TOLERANCE = 0.75
UNMATCHED_COST = TOLERANCE + 0.001
INVALID_COST = 1e6

MIN_PEAK_DISTANCE = int(1.5 * FS)


def match_events(reference, detected):
    reference = np.asarray(reference, dtype=float)
    detected = np.asarray(detected, dtype=float)

    n = len(reference)
    m = len(detected)

    if n == 0 or m == 0:
        return 0, np.array([])

    difference = np.abs(reference[:, None] - detected[None, :])
    valid = difference <= TOLERANCE

    cost = np.full((n + m, m + n), INVALID_COST)
    cost[:n, :m] = np.where(valid, difference, INVALID_COST)

    for i in range(n):
        cost[i, m + i] = UNMATCHED_COST

    for j in range(m):
        cost[n + j, j] = UNMATCHED_COST

    cost[n:, m:] = 0

    rows, cols = linear_sum_assignment(cost)

    matched_differences = []

    for row, col in zip(rows, cols):
        if row < n and col < m and valid[row, col]:
            matched_differences.append(
                reference[row] - detected[col]
            )

    return len(matched_differences), np.asarray(matched_differences)


def f1_score(count_a, count_b, matched):
    precision = matched / count_a if count_a else 0
    recall = matched / count_b if count_b else 0

    if precision + recall == 0:
        return 0

    return 200 * precision * recall / (precision + recall)


def detect_phase(signal, time, negative=False):
    detection_signal = -signal if negative else signal

    peaks, _ = find_peaks(
        detection_signal,
        distance=MIN_PEAK_DISTANCE,
        prominence=0.02
    )

    return time[peaks]


def main():
    results = []

    signal_files = sorted(
        DATA_DIR.rglob("bidmc_*_Signals.csv")
    )

    for signal_file in signal_files:

        recording = int(signal_file.stem.split("_")[1])

        breath_file = signal_file.with_name(
            f"bidmc_{recording:02d}_Breaths.csv"
        )

        if not breath_file.exists():
            continue

        signals = pd.read_csv(signal_file)
        annotations = pd.read_csv(breath_file)

        time = signals["Time [s]"].to_numpy()
        resp = signals[" RESP"].to_numpy()

        ann1 = (
            annotations.iloc[:, 0]
            .dropna().to_numpy(dtype=float) / FS
        )

        ann2 = (
            annotations.iloc[:, 1]
            .dropna().to_numpy(dtype=float) / FS
        )

        sos = butter(
            4, [0.1, 0.7],
            btype="bandpass",
            fs=FS,
            output="sos"
        )

        filtered = sosfiltfilt(sos, resp)

        positive_peaks = detect_phase(
            filtered, time, negative=False
        )

        negative_troughs = detect_phase(
            filtered, time, negative=True
        )

        # Determine which phase best represents each annotator.
        for annotator, reference in [
            (1, ann1),
            (2, ann2)
        ]:

            positive_matches, positive_errors = match_events(
                reference, positive_peaks
            )

            negative_matches, negative_errors = match_events(
                reference, negative_troughs
            )

            positive_f1 = f1_score(
                len(reference),
                len(positive_peaks),
                positive_matches
            )

            negative_f1 = f1_score(
                len(reference),
                len(negative_troughs),
                negative_matches
            )

            if negative_f1 > positive_f1:
                selected_phase = "negative_trough"
                selected_matches = negative_matches
                selected_errors = negative_errors
                selected_f1 = negative_f1
            else:
                selected_phase = "positive_peak"
                selected_matches = positive_matches
                selected_errors = positive_errors
                selected_f1 = positive_f1

            results.append({
                "recording": recording,
                "annotator": annotator,
                "selected_phase": selected_phase,
                "positive_f1": positive_f1,
                "negative_f1": negative_f1,
                "phase_aware_f1": selected_f1,
                "matched_events": selected_matches,
                "reference_events": len(reference),
                "mean_timing_error_s": (
                    np.mean(np.abs(selected_errors))
                    if len(selected_errors) else np.nan
                )
            })

    report = pd.DataFrame(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "phase_aware_annotation_audit.csv"
    report.to_csv(output_file, index=False)

    print("=" * 85)
    print("PHASE-AWARE ANNOTATION AUDIT")
    print("=" * 85)

    for annotator in [1, 2]:

        subset = report[report["annotator"] == annotator]

        print(f"\nANNOTATOR {annotator}")
        print("-" * 70)

        print(
            "Recordings analyzed:",
            len(subset)
        )

        print(
            "Positive-phase mean F1:",
            round(subset["positive_f1"].mean(), 3)
        )

        print(
            "Negative-phase mean F1:",
            round(subset["negative_f1"].mean(), 3)
        )

        print(
            "Phase-aware mean F1:",
            round(subset["phase_aware_f1"].mean(), 3)
        )

        print(
            "\nRecordings selecting negative phase:"
        )

        print(
            subset[subset["selected_phase"] == "negative_trough"][
                ["recording", "positive_f1", "negative_f1"]
            ].to_string(index=False)
        )

    print("\nRECORDING 44 CHECK")
    print(
        report[report["recording"] == 44][
            [
                "annotator",
                "selected_phase",
                "positive_f1",
                "negative_f1",
                "phase_aware_f1"
            ]
        ].to_string(index=False)
    )

    print("\nReport saved:", output_file)


if __name__ == "__main__":
    main()
