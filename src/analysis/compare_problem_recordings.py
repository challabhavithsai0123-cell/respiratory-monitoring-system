
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
RECORDINGS = [5, 13, 43, 44]
DURATION = 120

sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)

fig, axes = plt.subplots(
    len(RECORDINGS), 1,
    figsize=(16, 14)
)

for ax, number in zip(axes, RECORDINGS):

    prefix = f"bidmc_{number:02d}"

    signal_file = next(DATA_DIR.rglob(prefix + "_Signals.csv"))
    breath_file = next(DATA_DIR.rglob(prefix + "_Breaths.csv"))

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

    mask = time <= DURATION

    ax.plot(
        time[mask],
        filtered[mask],
        label="Filtered RESP",
        linewidth=1
    )

    visible_peaks = peaks[time[peaks] <= DURATION]

    ax.scatter(
        time[visible_peaks],
        filtered[visible_peaks],
        marker="x",
        color="red",
        label="Detected peaks",
        zorder=3
    )

    for index, color, label in [
        (0, "green", "Annotator 1"),
        (1, "orange", "Annotator 2")
    ]:

        annotation_times = (
            annotations.iloc[:, index].dropna().to_numpy() / FS
        )

        annotation_times = annotation_times[
            annotation_times <= DURATION
        ]

        annotation_amplitudes = np.interp(
            annotation_times,
            time,
            filtered
        )

        ax.scatter(
            annotation_times,
            annotation_amplitudes,
            marker="o",
            facecolors="none",
            edgecolors=color,
            label=label,
            s=45
        )

    ax.set_title(f"BIDMC Recording {number:02d}")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, DURATION)
    ax.grid(True)
    ax.legend(loc="upper right", ncol=4)

axes[-1].set_xlabel("Time (seconds)")

plt.tight_layout()

output_file = OUTPUT_DIR / "problem_recording_comparison.png"
plt.savefig(output_file, dpi=300)

print("Diagnostic comparison saved:", output_file)

plt.show()
