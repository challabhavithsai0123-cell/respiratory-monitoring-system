
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = ROOT / "results" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 125
WINDOW_SECONDS = 10
STEP_SECONDS = 5

RECORDINGS = [13, 15, 27, 46]

sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)

results = []

for number in RECORDINGS:

    file = next(
        DATA_DIR.rglob(f"bidmc_{number:02d}_Signals.csv")
    )

    df = pd.read_csv(file)

    time = df["Time [s]"].to_numpy()
    signal = df[" RESP"].to_numpy()

    filtered = sosfiltfilt(sos, signal)

    window = WINDOW_SECONDS * FS
    step = STEP_SECONDS * FS

    for start in range(0, len(filtered) - window + 1, step):

        segment = filtered[start:start + window]

        rms = np.sqrt(np.mean(segment ** 2))
        peak_to_peak = np.ptp(segment)
        max_abs = np.max(np.abs(segment))

        results.append({
            "recording": f"bidmc_{number:02d}",
            "start_time_s": time[start],
            "end_time_s": time[start + window - 1],
            "rms_amplitude": rms,
            "peak_to_peak": peak_to_peak,
            "maximum_absolute_amplitude": max_abs
        })

report = pd.DataFrame(results)

for number in RECORDINGS:

    subset = report[
        report["recording"] == f"bidmc_{number:02d}"
    ]

    print(f"\nRECORDING {number}")
    print("-" * 65)

    print(
        subset.nlargest(5, "peak_to_peak")[
            ["start_time_s", "end_time_s",
             "rms_amplitude", "peak_to_peak"]
        ].round(4).to_string(index=False)
    )

output = OUTPUT_DIR / "signal_quality_audit.csv"
report.to_csv(output, index=False)

print("\nSaved:", output)
