
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = ROOT / "results" / "graphs" / "configuration_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 125
PROMINENCE = 0.08
HEIGHT = 0.02
MIN_DISTANCE = int(1.5 * FS)

recordings = [13, 27, 46, 15]

sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)

for number in recordings:

    signal_file = next(
        DATA_DIR.rglob(f"bidmc_{number:02d}_Signals.csv")
    )

    breath_file = next(
        DATA_DIR.rglob(f"bidmc_{number:02d}_Breaths.csv")
    )

    signals = pd.read_csv(signal_file)
    annotations = pd.read_csv(breath_file)

    time = signals["Time [s]"].to_numpy()
    resp = signals[" RESP"].to_numpy()

    filtered = sosfiltfilt(sos, resp)

    positive_peaks, _ = find_peaks(
        filtered,
        distance=MIN_DISTANCE,
        prominence=PROMINENCE,
        height=HEIGHT
    )

    negative_peaks, _ = find_peaks(
        -filtered,
        distance=MIN_DISTANCE,
        prominence=PROMINENCE,
        height=HEIGHT
    )

    plt.figure(figsize=(16, 6))

    plt.plot(
        time,
        filtered,
        label="Filtered respiration",
        linewidth=0.8
    )

    plt.scatter(
        time[positive_peaks],
        filtered[positive_peaks],
        color="red",
        marker="x",
        label="Detected positive peaks",
        s=25
    )

    plt.scatter(
        time[negative_peaks],
        filtered[negative_peaks],
        color="blue",
        marker="x",
        label="Negative troughs",
        s=25
    )

    for column, color, label in [
        (annotations.columns[0], "green", "Annotator 1"),
        (annotations.columns[1], "magenta", "Annotator 2")
    ]:

        indices = pd.to_numeric(
            annotations[column],
            errors="coerce"
        ).dropna().to_numpy(dtype=int)

        indices = indices[
            (indices >= 0) & (indices < len(time))
        ]

        plt.scatter(
            time[indices],
            filtered[indices],
            color=color,
            facecolors="none",
            edgecolors=color,
            label=label,
            s=22
        )

    plt.title(
        f"BIDMC Recording {number:02d} — Configuration B"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Filtered amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"recording_{number:02d}_B.png"

    plt.savefig(output_file, dpi=160)
    plt.close()

    print("Saved:", output_file)

print("\nAll diagnostic plots generated.")
