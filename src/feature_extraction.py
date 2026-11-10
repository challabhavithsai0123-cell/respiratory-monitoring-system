
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from scipy.stats import skew, kurtosis


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FS = 125
WINDOW_SECONDS = 30
HOP_SECONDS = 15

PROMINENCE = 0.08
HEIGHT = 0.02
MIN_BREATH_DISTANCE = 1.5

DATA_DIR = Path("data/raw/BIDMC/bidmc-1.0.0/bidmc_csv")
OUTPUT_DIR = Path("results/reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# SIGNAL FILTERING
# --------------------------------------------------

def filter_respiration(signal):

    sos = butter(
        4,
        [0.1, 0.7],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


# --------------------------------------------------
# SPECTRAL FEATURES
# --------------------------------------------------

def calculate_spectral_features(signal):

    frequencies, power = welch(
        signal,
        fs=FS,
        nperseg=min(len(signal), FS * 10)
    )

    mask = (frequencies >= 0.1) & (frequencies <= 0.7)

    frequencies = frequencies[mask]
    power = power[mask]

    if len(power) == 0 or np.sum(power) == 0:
        return 0.0, 0.0, 0.0

    dominant_frequency = frequencies[np.argmax(power)]

    total_power = np.sum(power)

    probabilities = power / total_power

    spectral_entropy = -np.sum(
        probabilities * np.log2(probabilities + 1e-12)
    )

    band_power = np.trapezoid(power, frequencies)

    return dominant_frequency, spectral_entropy, band_power


# --------------------------------------------------
# MAIN FEATURE EXTRACTION
# --------------------------------------------------

def extract_features():

    all_features = []

    window_samples = WINDOW_SECONDS * FS
    hop_samples = HOP_SECONDS * FS

    signal_files = sorted(DATA_DIR.glob("bidmc_*_Signals.csv"))

    print("Recordings found:", len(signal_files))

    for recording_number, signal_file in enumerate(
        signal_files, start=1
    ):

        recording_id = signal_file.stem.replace(
            "_Signals", ""
        )

        print(
            f"Processing {recording_number}/{len(signal_files)}: "
            f"{recording_id}"
        )

        signals = pd.read_csv(signal_file)

        respiration = signals[" RESP"].to_numpy(dtype=float)

        if not np.all(np.isfinite(respiration)):
            raise ValueError(
                f"Invalid signal values in {recording_id}"
            )

        filtered = filter_respiration(respiration)

        # Detect positive respiratory peaks over the full recording.
        peaks, _ = find_peaks(
            filtered,
            distance=int(MIN_BREATH_DISTANCE * FS),
            prominence=PROMINENCE,
            height=HEIGHT
        )

        peak_times = peaks / FS

        # Load manual breath annotations.
        breath_file = DATA_DIR / f"{recording_id}_Breaths.csv"
        breaths = pd.read_csv(breath_file)

        ann1 = (
            breaths.iloc[:, 0].dropna().to_numpy(dtype=float) / FS
        )

        ann2 = (
            breaths.iloc[:, 1].dropna().to_numpy(dtype=float) / FS
        )

        # Extract features from overlapping windows.
        for start_sample in range(
            0,
            len(filtered) - window_samples + 1,
            hop_samples
        ):

            end_sample = start_sample + window_samples

            start_time = start_sample / FS
            end_time = end_sample / FS

            segment = filtered[start_sample:end_sample]

            if len(segment) != window_samples:
                continue

            window_peaks = peak_times[
                (peak_times >= start_time) &
                (peak_times < end_time)
            ]

            # Manual reference events in this window.
            reference_ann1 = ann1[
                (ann1 >= start_time) & (ann1 < end_time)
            ]

            reference_ann2 = ann2[
                (ann2 >= start_time) & (ann2 < end_time)
            ]

            # Time-domain features.
            signal_mean = np.mean(segment)
            signal_std = np.std(segment)
            signal_rms = np.sqrt(np.mean(segment ** 2))
            signal_range = np.ptp(segment)

            signal_skewness = skew(segment)
            signal_kurtosis = kurtosis(segment)

            # Zero-crossing rate.
            centered = segment - signal_mean

            zero_crossings = np.sum(
                centered[:-1] * centered[1:] < 0
            )

            zero_crossing_rate = (
                zero_crossings / len(segment)
            )

            # Respiratory rate from detected peaks.
            detected_count = len(window_peaks)

            detected_rr = (
                detected_count * 60.0 / WINDOW_SECONDS
            )

            # Inter-breath interval statistics.
            intervals = np.diff(window_peaks)

            mean_ibi = (
                np.mean(intervals) if len(intervals) > 0 else 0.0
            )

            std_ibi = (
                np.std(intervals) if len(intervals) > 0 else 0.0
            )

            # Spectral features.
            dominant_frequency, spectral_entropy, band_power = (
                calculate_spectral_features(segment)
            )

            # Manual reference respiratory rates.
            reference_rr_ann1 = (
                len(reference_ann1) * 60.0 / WINDOW_SECONDS
            )

            reference_rr_ann2 = (
                len(reference_ann2) * 60.0 / WINDOW_SECONDS
            )

            all_features.append({

                "recording_id": recording_id,
                "window_start_s": start_time,
                "window_end_s": end_time,

                "mean": signal_mean,
                "std": signal_std,
                "rms": signal_rms,
                "amplitude_range": signal_range,
                "skewness": signal_skewness,
                "kurtosis": signal_kurtosis,
                "zero_crossing_rate": zero_crossing_rate,

                "detected_breath_count": detected_count,
                "detected_rr_bpm": detected_rr,
                "mean_ibi_s": mean_ibi,
                "std_ibi_s": std_ibi,

                "dominant_frequency_hz": dominant_frequency,
                "spectral_entropy": spectral_entropy,
                "respiratory_band_power": band_power,

                "reference_rr_ann1_bpm": reference_rr_ann1,
                "reference_rr_ann2_bpm": reference_rr_ann2,

                "annotation_rr_difference_bpm": abs(
                    reference_rr_ann1 - reference_rr_ann2
                )
            })

    # Save consolidated feature dataset.
    feature_df = pd.DataFrame(all_features)

    output_file = OUTPUT_DIR / "respiratory_features.csv"

    feature_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 60)

    print("Total recordings:", feature_df["recording_id"].nunique())
    print("Total windows:", len(feature_df))
    print("Total features:", len(feature_df.columns))

    print("\nDataset shape:", feature_df.shape)

    print("\nMissing values:")
    print(feature_df.isnull().sum().sum())

    print("\nFirst five rows:")
    print(feature_df.head().to_string(index=False))

    print("\nSaved to:", output_file)


if __name__ == "__main__":
    extract_features()
