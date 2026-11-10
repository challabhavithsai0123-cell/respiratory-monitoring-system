
from pathlib import Path
import re

import numpy as np
import pandas as pd
import wfdb


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "BIDMC"
OUTPUT_DIR = PROJECT_ROOT / "results" / "reports"

WINDOW_SECONDS = 30
POSITIVE_LIMIT = 32767
CLIPPING_THRESHOLD_PERCENT = 5.0


# --------------------------------------------------
# Helper: longest consecutive run
# --------------------------------------------------

def longest_consecutive_run(mask):
    if len(mask) == 0 or not np.any(mask):
        return 0

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return int(np.max(ends - starts))


# --------------------------------------------------
# Main analysis
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = sorted(
        header
        for header in DATA_ROOT.rglob("*.hea")
        if re.fullmatch(r"bidmc\d{2}", header.stem.lower())
    )

    print(f"Main waveform records found: {len(headers)}")

    results = []
    processed_records = 0
    failed_records = []

    for header in headers:

        record_name = header.stem

        try:
            record = wfdb.rdrecord(
                str(header.with_suffix("")),
                physical=False
            )

            channel_names = [
                str(name).strip().strip(",").upper()
                for name in record.sig_name
            ]

            if "RESP" not in channel_names:
                print(f"RESP channel missing: {record_name}")
                failed_records.append(record_name)
                continue

            resp_index = channel_names.index("RESP")
            signal = np.asarray(record.d_signal[:, resp_index])

            fs = float(record.fs)
            window_size = int(WINDOW_SECONDS * fs)

            if window_size <= 0:
                raise ValueError("Invalid sampling frequency")

            complete_windows = len(signal) // window_size

            for window_index in range(complete_windows):

                start = window_index * window_size
                end = start + window_size

                segment = signal[start:end]

                positive_limit_mask = segment == POSITIVE_LIMIT

                positive_limit_percent = (
                    np.count_nonzero(positive_limit_mask)
                    / len(segment)
                ) * 100

                longest_run_samples = longest_consecutive_run(
                    positive_limit_mask
                )

                longest_run_seconds = (
                    longest_run_samples / fs
                )

                flagged = (
                    positive_limit_percent
                    > CLIPPING_THRESHOLD_PERCENT
                )

                results.append({
                    "recording": record_name,
                    "window_index": window_index,
                    "start_time_s": start / fs,
                    "end_time_s": end / fs,
                    "positive_limit_samples": int(
                        np.count_nonzero(positive_limit_mask)
                    ),
                    "positive_limit_percent": positive_limit_percent,
                    "longest_positive_limit_run_s": longest_run_seconds,
                    "flagged": flagged
                })

            processed_records += 1

            print(
                f"Processed {record_name}: "
                f"{complete_windows} windows"
            )

        except Exception as error:
            print(f"ERROR in {record_name}: {error}")
            failed_records.append(record_name)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    columns = [
        "recording",
        "window_index",
        "start_time_s",
        "end_time_s",
        "positive_limit_samples",
        "positive_limit_percent",
        "longest_positive_limit_run_s",
        "flagged"
    ]

    df = pd.DataFrame(results, columns=columns)

    output_file = OUTPUT_DIR / "window_saturation_analysis.csv"
    df.to_csv(output_file, index=False)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    flagged_df = df[df["flagged"] == True]

    print("\n========== ANALYSIS SUMMARY ==========")
    print(f"Records found: {len(headers)}")
    print(f"Records processed: {processed_records}")
    print(f"Records failed: {len(failed_records)}")
    print(f"Total complete windows: {len(df)}")
    print(f"Flagged windows: {len(flagged_df)}")

    if len(df) > 0:
        print(
            "Flagged window percentage: "
            f"{len(flagged_df) / len(df) * 100:.2f}%"
        )

    if len(flagged_df) > 0:

        print("\nFlagged windows by recording:")

        counts = (
            flagged_df.groupby("recording")
            .size()
            .sort_values(ascending=False)
        )

        print(counts.to_string())

        print("\nTop 15 windows by positive digital-limit occupancy:")

        top_windows = flagged_df.sort_values(
            "positive_limit_percent",
            ascending=False
        ).head(15)

        print(top_windows.to_string(index=False))

    else:
        print("\nNo windows exceeded the provisional threshold.")

    if failed_records:
        print("\nFailed records:", failed_records)

    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
