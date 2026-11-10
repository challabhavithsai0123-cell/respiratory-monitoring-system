
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = ROOT / "results" / "graphs"

FS = 125
TOLERANCE = 0.75

signal_file = next(
    DATA_DIR.rglob("bidmc_39_Signals.csv")
)

breath_file = signal_file.with_name(
    "bidmc_39_Breaths.csv"
)

signals = pd.read_csv(signal_file)
annotations = pd.read_csv(breath_file)

time = signals["Time [s]"].to_numpy()
resp = signals[" RESP"].to_numpy()

sos = butter(
    4, [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)

filtered = sosfiltfilt(sos, resp)

peaks_008, properties = find_peaks(
    filtered,
    distance=int(1.5 * FS),
    prominence=0.08
)

peaks_012, _ = find_peaks(
    filtered,
    distance=int(1.5 * FS),
    prominence=0.12
)

column = "breaths ann1 [signal sample no]"

reference = (
    annotations[column]
    .dropna()
    .astype(int)
    .to_numpy()
)

distances = (
    np.abs(reference[:, None] - peaks_008[None, :]) / FS
)

nr = len(reference)
nd = len(peaks_008)

size = nr + nd
cost = np.full((size, size), 1e6)

valid = distances <= TOLERANCE

cost[:nr, :nd] = np.where(valid, distances, 1e6)
cost[:nr, nd:] = TOLERANCE / 2
cost[nr:, :nd] = TOLERANCE / 2
cost[nr:, nd:] = 0

rows, cols = linear_sum_assignment(cost)

matched_indices = {
    c for r, c in zip(rows, cols)
    if r < nr and c < nd and valid[r, c]
}

false_indices = [
    i for i in range(nd)
    if i not in matched_indices
]

false_peaks = peaks_008[false_indices]

print("\nRECORDING 39 — FALSE POSITIVE ANALYSIS")
print("=" * 65)

print("Detected at 0.08:", len(peaks_008))
print("Detected at 0.12:", len(peaks_012))
print("Reference events:", len(reference))
print("Matched at 0.08:", len(matched_indices))
print("Unmatched detections at 0.08:", len(false_peaks))

print("\nFirst 20 unmatched detections:")

for index in false_peaks[:20]:
    print(
        f"Time: {time[index]:.3f} s | "
        f"Amplitude: {filtered[index]:.4f}"
    )

# Plot unmatched detections over the full signal.
plt.figure(figsize=(16, 6))

plt.plot(
    time,
    filtered,
    linewidth=0.8,
    label="Filtered respiratory signal"
)

plt.scatter(
    time[false_peaks],
    filtered[false_peaks],
    color="red",
    marker="x",
    s=45,
    label="Unmatched detections"
)

plt.title("Recording 39 — Unmatched Detections at Prominence 0.08")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.legend()

output = OUTPUT_DIR / "recording_39_false_positives.png"

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved:", output)
