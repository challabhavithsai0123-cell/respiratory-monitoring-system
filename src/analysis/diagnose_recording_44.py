import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"

FS = 125

signals = pd.read_csv(
    next(DATA_DIR.rglob("bidmc_44_Signals.csv"))
)

annotations = pd.read_csv(
    next(DATA_DIR.rglob("bidmc_44_Breaths.csv"))
)

resp = signals[" RESP"].to_numpy()

sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)

filtered = sosfiltfilt(sos, resp)

# Detect positive peaks and negative troughs
positive_peaks, _ = find_peaks(
    filtered,
    distance=int(1.5 * FS),
    prominence=0.12
)

negative_troughs, _ = find_peaks(
    -filtered,
    distance=int(1.5 * FS),
    prominence=0.12
)

positive_times = positive_peaks / FS
negative_times = negative_troughs / FS


def evaluate_alignment(reference, detected):
    if len(reference) == 0 or len(detected) == 0:
        return np.nan, 0

    errors = [
        np.min(np.abs(detected - t))
        for t in reference
    ]

    return np.median(errors), np.mean(
        np.array(errors) <= 0.75
    ) * 100


print("=" * 65)
print("RECORDING 44: PEAK AND TROUGH ALIGNMENT")
print("=" * 65)

for index, name in enumerate(["Annotator 1", "Annotator 2"]):

    reference = (
        annotations.iloc[:, index]
        .dropna()
        .to_numpy() / FS
    )

    peak_error, peak_match = evaluate_alignment(
        reference, positive_times
    )

    trough_error, trough_match = evaluate_alignment(
        reference, negative_times
    )

    print(f"\n{name}")
    print("Reference events:", len(reference))

    print("\nPositive peak alignment:")
    print("Median timing error:", round(peak_error, 3), "seconds")
    print("Events within 0.75 s:", round(peak_match, 2), "%")

    print("\nNegative trough alignment:")
    print("Median timing error:", round(trough_error, 3), "seconds")
    print("Events within 0.75 s:", round(trough_match, 2), "%")

print("\nDiagnostic completed.")