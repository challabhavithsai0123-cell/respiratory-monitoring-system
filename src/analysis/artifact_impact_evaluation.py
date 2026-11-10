
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


def robust_threshold(values, multiplier=4.0):
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad < 1e-10:
        return median + 1e-10

    return median + multiplier * 1.4826 * mad


def analyze_recording(recording_number):
    signal_file = next(
        DATA_DIR.rglob(f"bidmc_{recording_number:02d}_Signals.csv")
    )

    breath_file = next(
        DATA_DIR.rglob(f"bidmc_{recording_number:02d}_Breaths.csv")
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

    window_size = WINDOW_SECONDS * FS
    step_size = STEP_SECONDS * FS

    windows = []

    for start in range(0, len(filtered) - window_size + 1, step_size):
        segment = filtered[start:start + window_size]

        rms = np.sqrt(np.mean(segment ** 2))
        peak_to_peak = np.ptp(segment)
        derivative_rms = np.sqrt(
            np.mean(np.diff(segment) ** 2)
        ) * FS

        windows.append({
            "start_sample": start,
            "end_sample": start + window_size,
            "start_time_s": time[start],
            "end_time_s": time[start + window_size - 1],
            "rms": rms,
            "peak_to_peak": peak_to_peak,
            "derivative_rms": derivative_rms
        })

    window_df = pd.DataFrame(windows)

    # Robust limits calculated separately for each recording.
    rms_limit = robust_threshold(window_df["rms"].to_numpy())
    derivative_limit = robust_threshold(
        window_df["derivative_rms"].to_numpy()
    )

    window_df["flagged"] = (
        (window_df["rms"] > rms_limit) &
        (window_df["derivative_rms"] > derivative_limit)
    )

    results = []

    for annotator, column in [
        (1, "breaths ann1 [signal sample no]"),
        (2, " breaths ann2 [signal sample no]")
    ]:
        events = (
            annotations[column]
            .dropna()
            .to_numpy(dtype=int)
        )

        total_events = len(events)
        flagged_events = 0

        for _, window in window_df[window_df["flagged"]].iterrows():
            inside = (
                (events >= window["start_sample"]) &
                (events < window["end_sample"])
            )
            flagged_events += np.sum(inside)

        results.append({
            "recording": recording_number,
            "annotator": annotator,
            "total_annotated_breaths": total_events,
            "breaths_in_flagged_windows": flagged_events,
            "flagged_breath_percentage": (
                100 * flagged_events / total_events
                if total_events else 0
            ),
            "flagged_windows": int(window_df["flagged"].sum()),
            "total_windows": len(window_df)
        })

    print(f"\nRecording {recording_number}")
    print(window_df[window_df["flagged"]][
        ["start_time_s", "end_time_s", "rms", "derivative_rms"]
    ].to_string(index=False))

    return results


all_results = []

for recording in RECORDINGS:
    all_results.extend(analyze_recording(recording))

report = pd.DataFrame(all_results)

output_file = OUTPUT_DIR / "artifact_impact_audit.csv"
report.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("ARTIFACT IMPACT AUDIT")
print("=" * 70)
print(report.to_string(index=False))
print(f"\nReport saved to: {output_file}")
