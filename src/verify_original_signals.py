
import numpy as np
import pandas as pd
import wfdb

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "raw" / "BIDMC"
OUTPUT = ROOT / "results" / "reports"
OUTPUT.mkdir(parents=True, exist_ok=True)

import re

headers = sorted(
    header for header in DATA_ROOT.rglob("*.hea")
    if re.fullmatch(r"bidmc\d{2}", header.stem.lower())
)

print("Header files found:", len(headers))

if not headers:
    raise FileNotFoundError(
        f"No BIDMC header files found inside: {DATA_ROOT}"
    )

results = []

for header in headers:

    record_path = header.with_suffix("")
    record_name = header.stem

    try:
        record = wfdb.rdrecord(
            str(record_path),
            physical=False
        )

        channels = [
            str(name).strip().strip(",").upper()
            for name in record.sig_name
        ]

        if "RESP" not in channels:
            continue

        index = channels.index("RESP")
        signal = record.d_signal[:, index]

        total = len(signal)

        positive_count = int(np.sum(signal == 32767))
        negative_count = int(np.sum(signal == -32768))

        results.append({
            "recording": record_name,
            "samples": total,
            "unique_values": len(np.unique(signal)),
            "digital_minimum": int(np.min(signal)),
            "digital_maximum": int(np.max(signal)),
            "positive_limit_count": positive_count,
            "positive_limit_percent": positive_count / total * 100,
            "negative_limit_count": negative_count,
            "negative_limit_percent": negative_count / total * 100,
            "adc_gain": record.adc_gain[index],
            "adc_baseline": record.baseline[index]
        })

        print(f"Processed {record_name}")

    except Exception as error:
        print(f"Error processing {record_name}: {error}")

report = pd.DataFrame(results)

if report.empty:
    raise RuntimeError(
        "No respiratory signals were processed. Check the header files and errors above."
    )

report = report.sort_values(
    "positive_limit_percent",
    ascending=False

)

output_file = OUTPUT / "full_dataset_saturation_audit.csv"
report.to_csv(output_file, index=False)

print("\n========== FULL DATASET SATURATION AUDIT ==========\n")
print(report.to_string(index=False))

print("\nRecordings processed:", len(report))
print(
    "Recordings with more than 5% positive-limit samples:",
    int((report["positive_limit_percent"] > 5).sum())
)

print("\nReport saved:", output_file)
