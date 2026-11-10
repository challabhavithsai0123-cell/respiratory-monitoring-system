import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"

FS = 125

sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)

print("=" * 70)
print("ANNOTATION POLARITY ANALYSIS - FULL BIDMC DATASET")
print("=" * 70)

print(f"{'RECORDING':20s} | {'ANN1 MEDIAN':12s} | {'ANN2 MEDIAN':12s}")
print("-" * 70)

for signal_file in sorted(DATA_DIR.rglob("*_Signals.csv")):

    breath_file = signal_file.with_name(
        signal_file.name.replace("_Signals.csv", "_Breaths.csv")
    )

    if not breath_file.exists():
        continue

    signals = pd.read_csv(signal_file)
    annotations = pd.read_csv(breath_file)

    respiratory_signal = signals[" RESP"].to_numpy()

    filtered = sosfiltfilt(sos, respiratory_signal)

    medians = []

    for column in annotations.columns[:2]:

        indices = (
            annotations[column]
            .dropna()
            .astype(int)
            .to_numpy()
        )

        indices = indices[
            (indices >= 0) & (indices < len(filtered))
        ]

        median_amplitude = (
            np.median(filtered[indices])
            if len(indices)
            else np.nan
        )

        medians.append(median_amplitude)

    print(
        f"{signal_file.stem:20s} | "
        f"{medians[0]:12.4f} | "
        f"{medians[1]:12.4f}"
    )

print("\nAnalysis completed.")