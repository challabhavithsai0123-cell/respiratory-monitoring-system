
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"

FS = 125

signal_file = next(DATA_DIR.rglob("bidmc_27_Signals.csv"))
breath_file = next(DATA_DIR.rglob("bidmc_27_Breaths.csv"))

signals = pd.read_csv(signal_file)
annotations = pd.read_csv(breath_file)

resp = signals[" RESP"].to_numpy()

sos = butter(
    4, [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)

filtered = sosfiltfilt(sos, resp)

# Identify positive local peaks for diagnostic purposes.
peaks, properties = find_peaks(
    filtered,
    distance=int(0.8 * FS),
    prominence=0.02
)

peak_times = peaks / FS

ann1 = (
    annotations["breaths ann1 [signal sample no]"]
    .dropna().to_numpy(dtype=int)
) / FS

ann2 = (
    annotations[" breaths ann2 [signal sample no]"]
    .dropna().to_numpy(dtype=int)
) / FS

START = 450
END = 480
TOLERANCE = 0.75


def inspect_events(events, label):
    print(f"\n{label}")
    print("-" * 85)

    rows = []

    for event in events:
        if not START <= event <= END:
            continue

        if len(peak_times) == 0:
            continue

        distances = np.abs(peak_times - event)
        nearest_index = np.argmin(distances)

        nearest_peak = peak_times[nearest_index]
        difference = nearest_peak - event
        absolute_difference = abs(difference)

        rows.append({
            "annotation_time_s": event,
            "nearest_peak_time_s": nearest_peak,
            "signed_difference_s": difference,
            "absolute_difference_s": absolute_difference,
            "within_tolerance": absolute_difference <= TOLERANCE,
            "signal_amplitude": filtered[
                int(round(event * FS))
            ]
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        print(result.round(4).to_string(index=False))

        print(
            "\nEvents within 0.75 seconds:",
            result["within_tolerance"].sum(),
            "/",
            len(result)
        )


inspect_events(ann1, "ANNOTATOR 1")
inspect_events(ann2, "ANNOTATOR 2")

print("\nPeak alignment diagnostic completed.")
