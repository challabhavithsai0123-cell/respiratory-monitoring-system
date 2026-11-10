
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = ROOT / "results" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 125

signal_file = next(DATA_DIR.rglob("bidmc_27_Signals.csv"))
breath_file = next(DATA_DIR.rglob("bidmc_27_Breaths.csv"))

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

ann1 = (
    annotations["breaths ann1 [signal sample no]"]
    .dropna().to_numpy(dtype=int)
)

ann2 = (
    annotations[" breaths ann2 [signal sample no]"]
    .dropna().to_numpy(dtype=int)
)

ann1_times = ann1 / FS
ann2_times = ann2 / FS

START = 450
END = 480

mask = (time >= START) & (time <= END)

plt.figure(figsize=(16, 7))

plt.plot(
    time[mask],
    filtered[mask],
    label="Filtered respiratory signal",
    linewidth=1
)

for t in ann1_times:
    if START <= t <= END:
        plt.axvline(
            t, color="green", linestyle="--",
            alpha=0.8, label="Annotator 1"
        )

for t in ann2_times:
    if START <= t <= END:
        plt.axvline(
            t, color="magenta", linestyle=":",
            alpha=0.8, label="Annotator 2"
        )

plt.title("BIDMC Recording 27 — Annotation Review (450–480 s)")
plt.xlabel("Time (seconds)")
plt.ylabel("Filtered respiratory amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()

output = OUTPUT_DIR / "recording_27_zoom_review.png"
plt.savefig(output, dpi=200)
plt.show()

print("Graph saved:", output)
