
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = ROOT / "results" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 125

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

fig, axes = plt.subplots(
    3, 1,
    figsize=(16, 12)
)

thresholds = [0.08, 0.12, 0.16]

for ax, prominence in zip(axes, thresholds):

    peaks, _ = find_peaks(
        filtered,
        distance=int(1.5 * FS),
        prominence=prominence
    )

    ax.plot(
        time,
        filtered,
        label="Filtered respiratory signal",
        linewidth=0.8
    )

    ax.scatter(
        time[peaks],
        filtered[peaks],
        color="red",
        marker="x",
        s=20,
        label=f"Detected peaks ({prominence})"
    )

    for column, label, color in [
        ("breaths ann1 [signal sample no]", "Annotator 1", "green"),
        (" breaths ann2 [signal sample no]", "Annotator 2", "purple")
    ]:

        indices = (
            annotations[column]
            .dropna()
            .astype(int)
            .to_numpy()
        )

        indices = indices[
            (indices >= 0) & (indices < len(filtered))
        ]

        ax.scatter(
            time[indices],
            filtered[indices],
            facecolors="none",
            edgecolors=color,
            s=18,
            label=label
        )

    ax.set_title(f"Recording 39 — Prominence {prominence}")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (seconds)")

plt.tight_layout()

output = OUTPUT_DIR / "recording_39_threshold_analysis.png"

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", output)
