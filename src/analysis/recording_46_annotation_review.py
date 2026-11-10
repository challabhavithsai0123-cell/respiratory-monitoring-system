
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/raw/BIDMC/bidmc-1.0.0/bidmc_csv")
OUTPUT_DIR = Path("results/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

signals = pd.read_csv(DATA_DIR / "bidmc_46_Signals.csv")
breaths = pd.read_csv(DATA_DIR / "bidmc_46_Breaths.csv")

time = signals["Time [s]"].to_numpy()
resp = signals[" RESP"].to_numpy()

ann1 = breaths.iloc[:, 0].dropna().to_numpy() / 125
ann2 = breaths.iloc[:, 1].dropna().to_numpy() / 125

fig, axes = plt.subplots(4, 1, figsize=(18, 14))

for i, ax in enumerate(axes):
    start = i * 120
    end = min(start + 120, time[-1])

    mask = (time >= start) & (time < end)

    ax.plot(time[mask], resp[mask], linewidth=0.8,
            label="Respiration signal")

    for t in ann1[(ann1 >= start) & (ann1 < end)]:
        ax.axvline(t, color="green", alpha=0.7, linewidth=0.9)

    for t in ann2[(ann2 >= start) & (ann2 < end)]:
        ax.axvline(t, color="red", alpha=0.7, linewidth=0.9)

    ax.set_title(f"Recording 46: {start}-{end} seconds")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "Recording 46 - Detailed Annotation Comparison",
    fontsize=15
)
plt.tight_layout()

output = OUTPUT_DIR / "recording_46_zoomed_annotation_review.png"
plt.savefig(output, dpi=200)
plt.show()

print("Annotator 1 events:", len(ann1))
print("Annotator 2 events:", len(ann2))
print("Saved:", output)
