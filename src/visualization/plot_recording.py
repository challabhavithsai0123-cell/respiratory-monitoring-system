
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = PROJECT_ROOT / "results" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Locate recording 01
signal_file = next(DATA_DIR.rglob("bidmc_01_Signals.csv"))
breath_file = next(DATA_DIR.rglob("bidmc_01_Breaths.csv"))

# Load real physiological signals
signals = pd.read_csv(signal_file)
breaths = pd.read_csv(breath_file)

sampling_frequency = 125

time = signals["Time [s]"].to_numpy()

# Display first 60 seconds
duration = 60
mask = time <= duration

# Convert annotation sample numbers into seconds
ann1 = breaths.iloc[:, 0].dropna().to_numpy() / sampling_frequency
ann2 = breaths.iloc[:, 1].dropna().to_numpy() / sampling_frequency

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Respiratory waveform
axes[0].plot(time[mask], signals.loc[mask, " RESP"], linewidth=0.8)
axes[0].scatter(
    ann1[ann1 <= duration],
    np.interp(ann1[ann1 <= duration], time, signals[" RESP"]),
    marker="x",
    label="Annotator 1"
)
axes[0].scatter(
    ann2[ann2 <= duration],
    np.interp(ann2[ann2 <= duration], time, signals[" RESP"]),
    marker="o",
    facecolors="none",
    label="Annotator 2"
)
axes[0].set_title("Respiratory Signal (RESP)")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid(True)

# PPG waveform
axes[1].plot(time[mask], signals.loc[mask, " PLETH"], linewidth=0.8)
axes[1].set_title("Photoplethysmography Signal (PLETH)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

# ECG Lead II
axes[2].plot(time[mask], signals.loc[mask, " II"], linewidth=0.8)
axes[2].set_title("ECG Signal (Lead II)")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_ylabel("Amplitude")
axes[2].grid(True)

plt.tight_layout()

output_file = OUTPUT_DIR / "bidmc_recording_01_signals.png"
plt.savefig(output_file, dpi=300)
print(f"Visualization saved to: {output_file}")

plt.show()
