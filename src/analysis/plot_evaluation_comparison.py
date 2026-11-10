
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "results" / "reports"
GRAPH_DIR = ROOT / "results" / "graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    REPORT_DIR / "evaluation_comparison.csv"
)

fig, axes = plt.subplots(
    2, 1,
    figsize=(16, 10),
    sharex=True
)

for ax, annotator in zip(axes, [1, 2]):
    subset = (
        df[df["annotator"] == annotator]
        .copy()
        .sort_values("recording")
    )

    x = range(len(subset))

    ax.plot(
        x,
        subset["f1_percent_standard"],
        marker="o",
        markersize=3,
        label="Standard evaluation"
    )

    ax.plot(
        x,
        subset["f1_percent_phase"],
        marker="x",
        markersize=4,
        linestyle="--",
        label="Phase-aware evaluation"
    )

    ax.set_title(f"Annotator {annotator}")
    ax.set_ylabel("F1 Score (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Highlight Recording 44.
    recording_44 = subset.index[
        subset["recording"] == "bidmc_44"
    ]

    if len(recording_44):
        position = subset.index.get_loc(recording_44[0])
        ax.axvline(
            position,
            linestyle=":",
            alpha=0.7
        )

        ax.annotate(
            "Recording 44",
            xy=(position, 5),
            xytext=(position - 4, 20),
            arrowprops={"arrowstyle": "->"}
        )

axes[-1].set_xticks(list(x))
axes[-1].set_xticklabels(
    subset["recording"].str.replace("bidmc_", ""),
    rotation=90
)
axes[-1].set_xlabel("BIDMC Recording Number")

fig.suptitle(
    "Standard vs Phase-Aware Respiratory Event Detection",
    fontsize=15
)

plt.tight_layout()

output = GRAPH_DIR / "standard_vs_phase_aware_f1.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print(f"Graph saved to: {output}")
