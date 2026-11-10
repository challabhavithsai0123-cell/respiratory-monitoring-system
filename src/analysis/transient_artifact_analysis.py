
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, sosfiltfilt

# --------------------------------------------------
# PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "BIDMC"
GRAPH_DIR = ROOT / "results" / "graphs" / "artifact_analysis"
REPORT_DIR = ROOT / "results" / "reports"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FS = 125
WINDOW_SECONDS = 10
STEP_SECONDS = 5

RECORDINGS = [13, 15, 27, 46]

# --------------------------------------------------
# RESPIRATORY FILTER
# --------------------------------------------------

sos = butter(
    4,
    [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)


def robust_z_score(values):
    """
    Calculate robust z-scores using median
    and median absolute deviation.
    """

    median = np.median(values)

    mad = np.median(np.abs(values - median))

    if mad < 1e-12:
        return np.zeros_like(values)

    return 0.6745 * (values - median) / mad


# --------------------------------------------------
# ANALYZE ONE RECORDING
# --------------------------------------------------

all_results = []

for recording_number in RECORDINGS:

    filename = f"bidmc_{recording_number:02d}_Signals.csv"

    matches = list(DATA_DIR.rglob(filename))

    if not matches:
        print(f"File not found: {filename}")
        continue

    print("\n" + "=" * 65)
    print(f"ANALYZING RECORDING {recording_number}")
    print("=" * 65)

    signals = pd.read_csv(matches[0])

    time = signals["Time [s]"].to_numpy()
    respiration = signals[" RESP"].to_numpy()

    # Filter respiratory signal
    filtered = sosfiltfilt(sos, respiration)

    # First derivative
    derivative = np.gradient(filtered, 1 / FS)

    window_samples = WINDOW_SECONDS * FS
    step_samples = STEP_SECONDS * FS

    window_results = []

    # --------------------------------------------------
    # SLIDING WINDOW ANALYSIS
    # --------------------------------------------------

    for start in range(
        0,
        len(filtered) - window_samples + 1,
        step_samples
    ):

        end = start + window_samples

        segment = filtered[start:end]
        derivative_segment = derivative[start:end]

        rms = np.sqrt(np.mean(segment ** 2))

        peak_to_peak = np.ptp(segment)

        derivative_rms = np.sqrt(
            np.mean(derivative_segment ** 2)
        )

        window_results.append({
            "recording": recording_number,
            "start_time_s": time[start],
            "end_time_s": time[end - 1],
            "rms_amplitude": rms,
            "peak_to_peak": peak_to_peak,
            "derivative_rms": derivative_rms
        })

    df = pd.DataFrame(window_results)

    # --------------------------------------------------
    # ROBUST OUTLIER ANALYSIS
    # --------------------------------------------------

    df["rms_robust_z"] = robust_z_score(
        df["rms_amplitude"].to_numpy()
    )

    df["p2p_robust_z"] = robust_z_score(
        df["peak_to_peak"].to_numpy()
    )

    df["derivative_robust_z"] = robust_z_score(
        df["derivative_rms"].to_numpy()
    )

    # A window is flagged when at least one metric
    # is unusually high relative to this recording.
    df["flagged"] = (
        (df["rms_robust_z"] > 3.5) |
        (df["p2p_robust_z"] > 3.5) |
        (df["derivative_robust_z"] > 3.5)
    )

    all_results.append(df)

    # --------------------------------------------------
    # DISPLAY TOP WINDOWS
    # --------------------------------------------------

    print("\nTop 5 windows by RMS amplitude:")

    print(
        df.nlargest(5, "rms_amplitude")[
            [
                "start_time_s",
                "end_time_s",
                "rms_amplitude",
                "peak_to_peak",
                "derivative_rms",
                "flagged"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nFlagged windows: {df['flagged'].sum()} "
        f"out of {len(df)}"
    )

    # --------------------------------------------------
    # PLOT SIGNAL AND FLAGGED WINDOWS
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(
        time,
        filtered,
        linewidth=0.8,
        label="Filtered respiratory signal"
    )

    first_flag_label = True

    for _, row in df[df["flagged"]].iterrows():

        ax.axvspan(
            row["start_time_s"],
            row["end_time_s"],
            alpha=0.2,
            color="red",
            label=(
                "Statistically unusual window"
                if first_flag_label else None
            )
        )

        first_flag_label = False

    ax.set_title(
        f"BIDMC Recording {recording_number} "
        "— Transient Signal Analysis"
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Filtered amplitude")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    graph_path = (
        GRAPH_DIR /
        f"recording_{recording_number:02d}_artifact_analysis.png"
    )

    plt.savefig(graph_path, dpi=200)
    plt.close()

    print(f"Graph saved: {graph_path}")


# --------------------------------------------------
# SAVE COMBINED RESULTS
# --------------------------------------------------

if all_results:

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    report_path = (
        REPORT_DIR /
        "transient_artifact_analysis.csv"
    )

    final_results.to_csv(report_path, index=False)

    print("\n" + "=" * 65)
    print("TRANSIENT ARTIFACT ANALYSIS COMPLETED")
    print("=" * 65)

    print(f"Total windows analyzed: {len(final_results)}")

    print(
        "Total statistically flagged windows:",
        int(final_results["flagged"].sum())
    )

    print(f"\nReport saved: {report_path}")
    print(f"Graphs saved: {GRAPH_DIR}")

else:

    print("No recordings were successfully analyzed.")
