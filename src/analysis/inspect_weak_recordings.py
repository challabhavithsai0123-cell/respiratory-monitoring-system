
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
RECORDINGS = ["05", "13", "27"]

sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)

fig, axes = plt.subplots(
    3, 1,
    figsize=(17, 13)
)

for ax, number in zip(axes, RECORDINGS):

    signal_file = next(
        DATA_DIR.rglob(f"bidmc_{number}_Signals.csv")
    )

    breath_file = signal_file.with_name(
        f"bidmc_{number}_Breaths.csv"
    )

    signals = pd.read_csv(signal_file)
    annotations = pd.read_csv(breath_file)

    time = signals["Time [s]"].to_numpy()
    resp = signals[" RESP"].to_numpy()

    filtered = sosfiltfilt(sos, resp)

    peaks, _ = find_peaks(
        filtered,
        distance=int(1.5 * FS),
        prominence=0.12
    )

    ax.plot(
        time,
        filtered,
        linewidth=0.8,
        label="Filtered respiratory signal"
    )

    ax.scatter(
        time[peaks],
        filtered[peaks],
        color="red",
        marker="x",
        s=25,
        label="Detected positive peaks",
        zorder=3
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

        # Ignore invalid indices, if any.
        indices = indices[
            (indices >= 0) & (indices < len(filtered))
        ]

        ax.scatter(
            time[indices],
            filtered[indices],
            color=color,
            marker="o",
            s=18,
            facecolors="none",
            label=label,
            zorder=4
        )

    ax.set_title(f"BIDMC Recording {number}")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "Respiratory Detection Error Analysis",
    fontsize=16
)

plt.tight_layout()

output = OUTPUT_DIR / "weak_recordings_inspection.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved graph: {output}")
