
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"

FS = 125
RECORDINGS = [5, 11, 13, 43, 44, 48]

sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)

print("\nPROBLEMATIC RECORDING DIAGNOSTICS")
print("=" * 65)

for number in RECORDINGS:

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

    detected = time[peaks]

    print(f"\nRECORDING {number:02d}")
    print("-" * 45)
    print("Duration:", round(time[-1], 2), "seconds")
    print("Detected peaks:", len(detected))
    print("Detected rate:", round(60 / np.mean(np.diff(detected)), 2)
          if len(detected) > 1 else "N/A")

    for i, column in enumerate(annotations.columns):

        reference = annotations[column].dropna().to_numpy() / FS

        ref_rate = (
            60 / np.mean(np.diff(reference))
            if len(reference) > 1 else np.nan
        )

        print(f"\nAnnotator {i + 1}:")
        print("Reference events:", len(reference))
        print("Reference rate:", round(ref_rate, 2))
        print("First 8 annotation times:",
              np.round(reference[:8], 2))

    print("First 8 detected times:",
          np.round(detected[:8], 2))
