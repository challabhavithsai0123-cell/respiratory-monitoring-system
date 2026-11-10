
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "raw" / "BIDMC"
OUTPUT = ROOT / "results" / "reports"
OUTPUT.mkdir(parents=True, exist_ok=True)

RECORDINGS = ["bidmc_05", "bidmc_13", "bidmc_27", "bidmc_46"]
FS = 125


def get_runs(mask):
    """Return consecutive True runs as (start, end, length)."""
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return [
        (int(start), int(end), int(end - start))
        for start, end in zip(starts, ends)
    ]


summary = []
run_details = []

for recording in RECORDINGS:

    matches = list(DATA_ROOT.rglob(f"{recording}_Signals.csv"))

    if not matches:
        print(f"Missing: {recording}")
        continue

    df = pd.read_csv(matches[0])
    df.columns = df.columns.str.strip()

    signal = pd.to_numeric(
        df["RESP"], errors="coerce"
    ).dropna().to_numpy(dtype=float)

    minimum = np.min(signal)
    maximum = np.max(signal)

    min_runs = get_runs(signal == minimum)
    max_runs = get_runs(signal == maximum)

    all_boundary_runs = [
        ("minimum", start, end, length)
        for start, end, length in min_runs
    ] + [
        ("maximum", start, end, length)
        for start, end, length in max_runs
    ]

    all_boundary_runs.sort(key=lambda item: item[1])

    summary.append({
        "recording": recording,
        "minimum": minimum,
        "maximum": maximum,
        "minimum_samples": int(np.sum(signal == minimum)),
        "maximum_samples": int(np.sum(signal == maximum)),
        "minimum_run_count": len(min_runs),
        "maximum_run_count": len(max_runs),
        "longest_minimum_run_s": (
            max((r[2] for r in min_runs), default=0) / FS
        ),
        "longest_maximum_run_s": (
            max((r[2] for r in max_runs), default=0) / FS
        ),
        "total_boundary_runs": len(all_boundary_runs)
    })

    # Store the ten longest boundary runs for each recording
    longest_runs = sorted(
        all_boundary_runs,
        key=lambda item: item[3],
        reverse=True
    )[:10]

    for boundary, start, end, length in longest_runs:
        run_details.append({
            "recording": recording,
            "boundary": boundary,
            "start_seconds": start / FS,
            "end_seconds": end / FS,
            "duration_seconds": length / FS,
            "sample_count": length
        })

summary_df = pd.DataFrame(summary)
details_df = pd.DataFrame(run_details)

summary_path = OUTPUT / "boundary_run_summary.csv"
details_path = OUTPUT / "boundary_run_details.csv"

summary_df.to_csv(summary_path, index=False)
details_df.to_csv(details_path, index=False)

print("\n========== BOUNDARY RUN SUMMARY ==========\n")
print(summary_df.to_string(index=False))

print("\n========== TEN LONGEST BOUNDARY RUNS ==========\n")
print(details_df.to_string(index=False))

print(f"\nSaved: {summary_path}")
print(f"Saved: {details_path}")
