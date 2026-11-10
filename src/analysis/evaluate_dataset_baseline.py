
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.optimize import linear_sum_assignment

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = ROOT / "results" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Signal-processing parameters
FS = 125
LOW_CUTOFF = 0.1
HIGH_CUTOFF = 0.7
FILTER_ORDER = 4

MIN_PEAK_DISTANCE = int(1.5 * FS)
PEAK_PROMINENCE = 0.12
MATCH_TOLERANCE = 0.75

# Find all signal recordings
signal_files = sorted(DATA_DIR.rglob("*_Signals.csv"))

results = []

def evaluate_events(detected, reference):
    """Perform one-to-one event matching."""

    if len(detected) == 0 or len(reference) == 0:
        tp = 0
        errors = []
    else:
        distances = np.abs(
            detected[:, None] - reference[None, :]
        )

        row_indices, col_indices = linear_sum_assignment(distances)

        valid = (
            distances[row_indices, col_indices]
            <= MATCH_TOLERANCE
        )

        errors = distances[row_indices[valid], col_indices[valid]]
        tp = len(errors)

    fp = len(detected) - tp
    fn = len(reference) - tp

    precision = tp / (tp + fp) * 100 if tp + fp else 0
    recall = tp / (tp + fn) * 100 if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    timing_error = np.mean(errors) if len(errors) else np.nan

    return tp, precision, recall, f1, timing_error


def calculate_rate(times):
    if len(times) < 2:
        return np.nan

    intervals = np.diff(times)

    return 60 / np.mean(intervals)


print("=" * 60)
print("BIDMC FULL DATASET EVALUATION")
print("=" * 60)

for signal_file in signal_files:

    breath_file = signal_file.with_name(
        signal_file.name.replace("_Signals.csv", "_Breaths.csv")
    )

    if not breath_file.exists():
        print("Missing annotation:", signal_file.name)
        continue

    try:
        signals = pd.read_csv(signal_file)
        annotations = pd.read_csv(breath_file)

        time = signals["Time [s]"].to_numpy()
        resp = signals[" RESP"].to_numpy()

        # Filter respiratory waveform
        sos = butter(
            FILTER_ORDER,
            [LOW_CUTOFF, HIGH_CUTOFF],
            btype="bandpass",
            fs=FS,
            output="sos"
        )

        filtered = sosfiltfilt(sos, resp)

        # Detect respiratory peaks
        peaks, _ = find_peaks(
            filtered,
            distance=MIN_PEAK_DISTANCE,
            prominence=PEAK_PROMINENCE
        )

        detected_times = time[peaks]

        detected_rate = calculate_rate(detected_times)

        # Evaluate against both annotators
        for annotator_index in range(2):

            reference = (
                annotations.iloc[:, annotator_index]
                .dropna()
                .to_numpy() / FS
            )

            tp, precision, recall, f1, timing_error = (
                evaluate_events(detected_times, reference)
            )

            reference_rate = calculate_rate(reference)

            rate_error = (
                abs(detected_rate - reference_rate)
                if np.isfinite(detected_rate)
                and np.isfinite(reference_rate)
                else np.nan
            )

            results.append({
                "recording": signal_file.name,
                "annotator": annotator_index + 1,
                "detected_events": len(detected_times),
                "reference_events": len(reference),
                "matched_events": tp,
                "precision_percent": precision,
                "recall_percent": recall,
                "f1_percent": f1,
                "detected_rate_bpm": detected_rate,
                "reference_rate_bpm": reference_rate,
                "rate_error_bpm": rate_error,
                "mean_timing_error_s": timing_error
            })

        print("Processed:", signal_file.name)

    except Exception as error:
        print("ERROR:", signal_file.name, error)

# Save detailed results
if not results:
    raise RuntimeError("No recordings were successfully evaluated.")

df = pd.DataFrame(results)

output_file = OUTPUT_DIR / "bidmc_evaluation_results.csv"
df.to_csv(output_file, index=False)

# Overall results for each annotator
print("\n" + "=" * 60)
print("OVERALL PERFORMANCE")
print("=" * 60)

for annotator in [1, 2]:

    subset = df[df["annotator"] == annotator]

    print(f"\nANNOTATOR {annotator}")
    print("Recordings evaluated:", len(subset))

    for metric in [
        "precision_percent",
        "recall_percent",
        "f1_percent",
        "rate_error_bpm",
        "mean_timing_error_s"
    ]:
        print(
            metric,
            ":",
            round(subset[metric].mean(), 3)
        )

print("\nDetailed results saved to:")
print(output_file)
