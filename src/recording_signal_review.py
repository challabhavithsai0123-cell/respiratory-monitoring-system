
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "raw" / "BIDMC"
OUTPUT = ROOT / "results" / "graphs"

OUTPUT.mkdir(parents=True, exist_ok=True)

RECORDINGS = ["bidmc_05", "bidmc_13", "bidmc_27", "bidmc_46"]

FS = 125
DURATION = 180  # First 3 minutes

fig, axes = plt.subplots(
    len(RECORDINGS), 1,
    figsize=(16, 14)
)

for ax, recording in zip(axes, RECORDINGS):

    signal_matches = list(DATA_ROOT.rglob(f"{recording}_Signals.csv"))
    breath_matches = list(DATA_ROOT.rglob(f"{recording}_Breaths.csv"))

    if not signal_matches or not breath_matches:
        raise FileNotFoundError(
            f"Could not find signal or breath file for {recording}"
        )

    signals = pd.read_csv(signal_matches[0])
    breaths = pd.read_csv(breath_matches[0])

    signals.columns = signals.columns.str.strip()
    breaths.columns = breaths.columns.str.strip()

    if "RESP" not in signals.columns:
        raise ValueError(f"RESP column missing in {recording}")

    time = signals["Time [s]"].to_numpy()
    resp = signals["RESP"].to_numpy()

    mask = time <= DURATION

    ax.plot(
        time[mask],
        resp[mask],
        linewidth=0.7,
        label="Raw RESP signal"
    )

    annotation_columns = [
        ("breaths ann1 [signal sample no]", "Annotator 1"),
        ("breaths ann2 [signal sample no]", "Annotator 2")
    ]

    for column, label in annotation_columns:

        if column not in breaths.columns:
            print(f"Warning: {column} missing in {recording}")
            continue

        samples = pd.to_numeric(
            breaths[column], errors="coerce"
        ).dropna().to_numpy()

        samples = samples.astype(int)

        samples = samples[
            (samples >= 0) &
            (samples < len(resp))
        ]

        annotation_times = samples / FS

        valid = annotation_times <= DURATION

        samples = samples[valid]
        annotation_times = annotation_times[valid]

        ax.scatter(
            annotation_times,
            resp[samples],
            s=28,
            marker="o" if label == "Annotator 1" else "x",
            label=label,
            zorder=3
        )

    ax.set_title(recording)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("RESP amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

fig.suptitle(
    "Respiratory Signal and Reference Breath Annotations",
    fontsize=15
)

fig.tight_layout()

output_file = OUTPUT / "difficult_recordings_waveform_review.png"

fig.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nWaveform review saved:")
print(output_file)
