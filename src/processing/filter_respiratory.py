
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, sosfiltfilt

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = PROJECT_ROOT / "results" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load recording 01
signal_file = next(DATA_DIR.rglob("bidmc_01_Signals.csv"))
data = pd.read_csv(signal_file)

# Extract respiratory signal
fs = 125
time = data["Time [s]"].to_numpy()
resp = data[" RESP"].to_numpy()

# Design Butterworth bandpass filter
low_cutoff = 0.1
high_cutoff = 0.7
order = 4

sos = butter(
    order,
    [low_cutoff, high_cutoff],
    btype="bandpass",
    fs=fs,
    output="sos"
)

# Apply zero-phase filtering
filtered_resp = sosfiltfilt(sos, resp)

# Plot first 60 seconds
mask = time <= 60

fig, axes = plt.subplots(2, 1, figsize=(15, 8))

axes[0].plot(time[mask], resp[mask])
axes[0].set_title("Original Respiratory Signal")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

axes[1].plot(time[mask], filtered_resp[mask])
axes[1].set_title("Filtered Respiratory Signal")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

plt.tight_layout()

output_file = OUTPUT_DIR / "respiratory_filter_comparison.png"
plt.savefig(output_file, dpi=300)

print("Filtering completed successfully.")
print("Sampling frequency:", fs, "Hz")
print("Filter range:", low_cutoff, "-", high_cutoff, "Hz")
print("Filter order:", order)
print("Output saved:", output_file)

plt.show()
