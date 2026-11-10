
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = ROOT / "results" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load real recording
signal_file = next(DATA_DIR.rglob("bidmc_01_Signals.csv"))
breath_file = next(DATA_DIR.rglob("bidmc_01_Breaths.csv"))

signals = pd.read_csv(signal_file)
annotations = pd.read_csv(breath_file)

# Sampling information
fs = 125
time = signals["Time [s]"].to_numpy()
resp = signals[" RESP"].to_numpy()

# Butterworth bandpass filtering
sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered = sosfiltfilt(sos, resp)

# Automatic respiratory peak detection
peaks, properties = find_peaks(
    filtered,
    distance=int(1.5 * fs),
    prominence=0.12
)

detected_times = time[peaks]

# Calculate respiratory rate
intervals = np.diff(detected_times)
respiratory_rate = 60 / np.mean(intervals)

# Compare detections with manual annotations
def compare_annotations(annotation_column):
    manual_times = (
        annotations[annotation_column].dropna().to_numpy() / fs
    )

    errors = []
    matched = 0

    for manual_time in manual_times:
        differences = np.abs(detected_times - manual_time)

        if len(differences) == 0:
            continue

        nearest = np.argmin(differences)

        if differences[nearest] <= 0.75:
            matched += 1
            errors.append(differences[nearest])

    precision_denominator = len(detected_times)
    recall = matched / len(manual_times) * 100
    precision = (
        matched / precision_denominator * 100
        if precision_denominator else 0
    )

    mean_error = np.mean(errors) if errors else float("nan")

    return matched, recall, precision, mean_error

print("=" * 45)
print("AUTOMATIC RESPIRATORY ANALYSIS")
print("=" * 45)

print("Recording: BIDMC 01")
print("Detected breathing peaks:", len(peaks))
print("Estimated respiratory rate:",
      round(respiratory_rate, 2), "breaths/min")

for column in annotations.columns:
    matched, recall, precision, error = compare_annotations(column)

    print("\n", column.strip())
    print("Matched annotations:", matched)
    print("Recall:", round(recall, 2), "%")
    print("Precision:", round(precision, 2), "%")
    print("Mean timing error:", round(error, 3), "seconds")

# Visualize first 60 seconds
mask = time <= 60

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    time[mask],
    filtered[mask],
    label="Filtered respiratory signal"
)

visible_peaks = peaks[time[peaks] <= 60]

ax.scatter(
    time[visible_peaks],
    filtered[visible_peaks],
    color="red",
    marker="x",
    s=55,
    label="Automatically detected peaks"
)

ax.set_title("Automatic Respiratory Cycle Detection")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True)

plt.tight_layout()

output_file = OUTPUT_DIR / "automatic_breath_detection.png"
plt.savefig(output_file, dpi=300)

print("\nDetection graph saved:", output_file)

plt.show()
