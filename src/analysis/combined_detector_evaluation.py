
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

ANNOTATORS = {
    1: "breaths ann1 [signal sample no]",
    2: " breaths ann2 [signal sample no]"
}

CONFIGURATIONS = {
    "A_original": {"prominence": 0.12, "height": None},
    "B_height_002": {"prominence": 0.08, "height": 0.02},
    "C_height_000": {"prominence": 0.08, "height": 0.00},
    "D_height_012": {"prominence": 0.12, "height": 0.02}
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


results = []

signal_files = sorted(
    DATA_DIR.rglob("bidmc_*_Signals.csv")
)

for signal_file in signal_files:

    recording = signal_file.stem.replace("_Signals", "")
    breath_file = signal_file.with_name(recording + "_Breaths.csv")

    if not breath_file.exists():
        continue

    signals = pd.read_csv(signal_file)
    annotations = pd.read_csv(breath_file)

    resp = signals[" RESP"].to_numpy()
    filtered = sosfiltfilt(sos, resp)

    for annotator, column in ANNOTATORS.items():

        reference = (
            annotations[column]
            .dropna()
            .astype(int)
            .to_numpy()
        )

        # Phase-aware handling of the confirmed exception.
        use_negative_phase = (
            recording == "bidmc_44" and annotator == 1
        )

        detection_signal = -filtered if use_negative_phase else filtered

        phase = (
            "negative_trough"
            if use_negative_phase
            else "positive_peak"
        )

        for config_name, settings in CONFIGURATIONS.items():

            kwargs = {}

            if settings["height"] is not None:
                kwargs["height"] = settings["height"]

            detected, _ = find_peaks(
                detection_signal,
                distance=MIN_DISTANCE,
                prominence=settings["prominence"],
                **kwargs
            )

            matched = evaluate_events(reference, detected)

            precision = (
                matched / len(detected) if len(detected) else 0
            )

            recall = (
                matched / len(reference) if len(reference) else 0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )

            results.append({
                "recording": recording,
                "annotator": annotator,
                "phase": phase,
                "configuration": config_name,
                "prominence": settings["prominence"],
                "height": (
                    "None"
                    if settings["height"] is None
                    else settings["height"]
                ),
                "detected_events": len(detected),
                "reference_events": len(reference),
                "matched_events": matched,
                "precision_percent": precision * 100,
                "recall_percent": recall * 100,
                "f1_percent": f1 * 100
            })

    print("Processed:", recording)

df = pd.DataFrame(results)

detail_path = OUTPUT_DIR / "combined_detector_details.csv"
df.to_csv(detail_path, index=False)

summary = (
    df.groupby(["annotator", "configuration"])
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

summary_path = OUTPUT_DIR / "combined_detector_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nCOMBINED DETECTOR EVALUATION")
print("=" * 90)

for annotator in [1, 2]:

    subset = summary[summary["annotator"] == annotator]

    print(f"\nANNOTATOR {annotator}")

    print(
        subset[
            [
                "configuration",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "pooled_f1"
            ]
        ].round(3).to_string(index=False)
    )

print("\nRECORDING 44 VERIFICATION")

print(
    df[
        (df["recording"] == "bidmc_44")
    ][
        [
            "annotator",
            "phase",
            "configuration",
            "detected_events",
            "matched_events",
            "f1_percent"
        ]
    ].round(3).to_string(index=False)
)

print("\nSaved detailed results:", detail_path)
print("Saved summary:", summary_path)
