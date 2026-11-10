
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.signal import butter, sosfiltfilt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Respiratory Monitoring System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parents[1]

REPORTS = ROOT / "results" / "reports"
GRAPHS = ROOT / "results" / "graphs"
MODELS = ROOT / "results" / "models"
RAW_DATA = ROOT / "data" / "deployment"

st.title("🫁 Intelligent Respiratory Monitoring System")
st.caption(
    "Respiratory signal processing, rate estimation, "
    "machine learning evaluation and signal-quality analysis"
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(filename):
    path = REPORTS / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_metadata():
    path = MODELS / "model_metadata.json"

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def find_signal_files():
    if not RAW_DATA.exists():
        return {}

    files = {}

    for path in RAW_DATA.rglob("*Signals.csv"):
        name = path.name.lower()

        if name.startswith("bidmc_") and name.endswith("_signals.csv"):
            recording = name.replace("bidmc_", "").replace(
                "_signals.csv", ""
            )
            files[recording] = str(path)

    return files


@st.cache_data
def read_signal(path):
    df = pd.read_csv(path)

    df.columns = [
        str(column).strip().upper()
        for column in df.columns
    ]

    return df


# ============================================================
# SIGNAL PROCESSING
# ============================================================

def filter_respiratory_signal(signal, fs=125):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 100:
        return signal

    if not np.all(np.isfinite(signal)):
        signal = pd.Series(signal).interpolate(
            limit_direction="both"
        ).to_numpy()

    sos = butter(
        4,
        [0.1, 0.7],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


# ============================================================
# LOAD PROJECT RESULTS
# ============================================================

model_comparison = load_csv("model_comparison.csv")
oof_metrics = load_csv("pooled_oof_metrics.csv")
oof_predictions = load_csv("pooled_oof_predictions.csv")
test_predictions = load_csv("model_test_predictions.csv")

feature_importance = load_csv(
    "extra_trees_feature_importance.csv"
)

quality_recordings = load_csv(
    "full_dataset_saturation_audit.csv"
)

quality_windows = load_csv(
    "window_saturation_analysis.csv"
)

quality_predictions = load_csv(
    "window_quality_prediction_analysis.csv"
)

metadata = load_metadata()
signal_files = find_signal_files()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Overview",
        "Signal Explorer",
        "Respiratory Analysis",
        "Model Evaluation",
        "Signal Quality",
        "Research Graphs"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Dataset: PhysioNet BIDMC PPG and Respiration"
)

st.sidebar.caption(
    "Sampling frequency: 125 Hz"
)

st.sidebar.caption(
    "Analysis type: Retrospective research"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Available Recordings",
        len(signal_files)
    )

    col2.metric(
        "OOF Windows",
        len(oof_predictions)
    )

    col3.metric(
        "Holdout Windows",
        len(test_predictions)
    )

    col4.metric(
        "Trained Model",
        "Available" if (MODELS / "best_respiratory_model.joblib").exists()
        else "Not found"
    )

    st.divider()

    st.subheader("Project Architecture")

    st.markdown("""
    **1. Dataset acquisition**

    PhysioNet BIDMC respiratory signals and breath annotations.

    **2. Signal processing**

    Respiratory signal filtering and waveform analysis.

    **3. Breath detection**

    Peak-based respiratory event detection and respiratory-rate estimation.

    **4. Feature extraction**

    Time-domain and frequency-domain respiratory features.

    **5. Machine learning**

    Random Forest, Extra Trees and Gradient Boosting regression models.

    **6. Evaluation**

    Group-based cross-validation, independent holdout testing,
    error analysis and signal-quality investigations.

    **7. Visualization**

    Interactive respiratory waveform and model evaluation dashboard.
    """)

    st.divider()

    st.subheader("Model Evaluation Summary")

    if not model_comparison.empty:
        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Model comparison report not found.")

    st.info(
        "This application displays retrospective analysis of recorded "
        "signals. It is not a medical diagnostic system."
    )


# ============================================================
# SIGNAL EXPLORER
# ============================================================

elif page == "Signal Explorer":

    st.header("Respiratory Signal Explorer")

    if not signal_files:
        st.error(
            "BIDMC signal files were not found. "
            "Check data/raw/BIDMC."
        )

    else:

        recording_ids = sorted(signal_files.keys())

        selected = st.selectbox(
            "Select BIDMC recording",
            recording_ids,
            index=0
        )

        df = read_signal(signal_files[selected])

        if "RESP" not in df.columns:
            st.error("RESP channel not found in selected CSV.")

        else:

            time_column = next(
                (
                    col for col in df.columns
                    if col.startswith("TIME")
                ),
                None
            )

            if time_column:
                time = df[time_column].to_numpy(dtype=float)
            else:
                time = np.arange(len(df)) / 125.0

            raw = df["RESP"].to_numpy(dtype=float)

            st.write(
                f"Recording: BIDMC {selected.upper()} | "
                f"Samples: {len(raw):,} | "
                f"Duration: {time[-1]:.1f} seconds"
            )

            max_time = max(1, int(time[-1]))

            duration = st.slider(
                "Display duration (seconds)",
                min_value=30,
                max_value=max(30, max_time),
                value=min(180, max(30, max_time)),
                step=30
            )

            mask = time <= duration

            filtered = filter_respiratory_signal(raw)

            col1, col2 = st.columns(2)

            col1.metric(
                "Raw Signal Mean",
                f"{np.mean(raw[mask]):.4f}"
            )

            col2.metric(
                "Filtered Signal Std",
                f"{np.std(filtered[mask]):.4f}"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=time[mask],
                    y=raw[mask],
                    name="Raw RESP",
                    line=dict(width=1)
                )
            )

            fig.update_layout(
                title="Original Respiratory Waveform",
                xaxis_title="Time (seconds)",
                yaxis_title="Signal Amplitude",
                template="plotly_white",
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=time[mask],
                    y=filtered[mask],
                    name="Filtered RESP",
                    line=dict(width=1.2)
                )
            )

            fig2.update_layout(
                title="Filtered Respiratory Waveform",
                xaxis_title="Time (seconds)",
                yaxis_title="Filtered Amplitude",
                template="plotly_white",
                height=400
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


# ============================================================
# RESPIRATORY ANALYSIS
# ============================================================

elif page == "Respiratory Analysis":

    st.header("Respiratory Rate Analysis")

    st.markdown("""
    This section displays respiratory-rate estimates already
    generated by the research pipeline. It does not independently
    retrain or modify the trained model.
    """)

    if oof_predictions.empty and test_predictions.empty:

        st.warning("Prediction reports are not available.")

    else:

        evaluation_type = st.radio(
            "Evaluation dataset",
            ["Holdout", "OOF"],
            horizontal=True
        )

        predictions = (
            test_predictions
            if evaluation_type == "Holdout"
            else oof_predictions
        )

        if not predictions.empty:

            recording_column = "recording_id"

            recording_ids = sorted(
                predictions[recording_column]
                .astype(str)
                .unique()
            )

            recording = st.selectbox(
                "Select recording",
                recording_ids
            )

            selected_data = predictions[
                predictions[recording_column].astype(str)
                == recording
            ].copy()

            st.subheader(f"Respiratory Estimates: {recording}")

            st.dataframe(
                selected_data,
                use_container_width=True,
                hide_index=True
            )

            fig = go.Figure()

            x = selected_data["window_start_s"]

            if "consensus_rr_bpm" in selected_data:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=selected_data["consensus_rr_bpm"],
                        name="Consensus Reference",
                        mode="lines+markers"
                    )
                )

            baseline_column = "signal_processing_baseline"

            if baseline_column in selected_data:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=selected_data[baseline_column],
                        name="Signal Processing Baseline",
                        mode="lines+markers"
                    )
                )

            model_columns = [
                "random_forest",
                "extra_trees",
                "gradient_boosting",
                "random_forest_prediction",
                "extra_trees_prediction",
                "gradient_boosting_prediction"
            ]

            for column in model_columns:

                if column in selected_data.columns:

                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=selected_data[column],
                            name=column.replace("_", " ").title(),
                            mode="lines"
                        )
                    )

            fig.update_layout(
                title="Respiratory Rate Across Analysis Windows",
                xaxis_title="Window Start Time (seconds)",
                yaxis_title="Respiratory Rate (BPM)",
                template="plotly_white",
                height=500
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.header("Machine Learning Evaluation")

    st.subheader("Model Comparison")

    if not model_comparison.empty:
        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Model comparison report unavailable.")

    st.subheader("Pooled Out-of-Fold Metrics")

    if not oof_metrics.empty:
        st.dataframe(
            oof_metrics,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("Feature Importance")

    if not feature_importance.empty:

        importance_column = next(
            (
                col for col in feature_importance.columns
                if "importance" in col.lower()
            ),
            None
        )

        feature_column = next(
            (
                col for col in feature_importance.columns
                if "feature" in col.lower()
            ),
            None
        )

        if importance_column and feature_column:

            plot_data = feature_importance.sort_values(
                importance_column
            )

            fig = go.Figure(
                go.Bar(
                    x=plot_data[importance_column],
                    y=plot_data[feature_column],
                    orientation="h"
                )
            )

            fig.update_layout(
                title="Extra Trees Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_white",
                height=550
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.dataframe(feature_importance)

    st.subheader("Model Metadata")

    if metadata:
        st.json(metadata)
    else:
        st.warning("Model metadata file unavailable.")


# ============================================================
# SIGNAL QUALITY
# ============================================================

elif page == "Signal Quality":

    st.header("Signal Quality Investigation")

    st.markdown("""
    Positive digital-limit occupancy is used as a statistical
    signal-quality indicator. It is consistent with positive-side
    clipping or saturation, but does not independently establish
    the physical acquisition mechanism.
    """)

    if not quality_recordings.empty:

        st.subheader("Recording-Level Audit")

        st.dataframe(
            quality_recordings,
            use_container_width=True,
            hide_index=True
        )

        percent_column = "positive_limit_percent"

        if percent_column in quality_recordings.columns:

            fig = go.Figure(
                go.Bar(
                    x=quality_recordings["recording"],
                    y=quality_recordings[percent_column],
                    name="Positive-limit occupancy"
                )
            )

            fig.update_layout(
                title="Positive Digital-Limit Occupancy by Recording",
                xaxis_title="Recording",
                yaxis_title="Occupancy (%)",
                template="plotly_white",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    if not quality_windows.empty:

        st.subheader("Window-Level Quality Audit")

        st.dataframe(
            quality_windows,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    if not quality_predictions.empty:

        st.subheader("Quality Versus Prediction Error")

        evaluation = st.selectbox(
            "Select evaluation",
            ["OOF", "Holdout"]
        )

        subset = quality_predictions[
            quality_predictions["evaluation"] == evaluation
        ]

        if not subset.empty:

            st.dataframe(
                subset,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# RESEARCH GRAPHS
# ============================================================

elif page == "Research Graphs":

    st.header("Research Visualization Gallery")

    if GRAPHS.exists():

        graph_files = sorted(
            [
                path for path in GRAPHS.rglob("*.png")
                if path.is_file()
            ]
        )

        if graph_files:

            selected_graph = st.selectbox(
                "Select research graph",
                graph_files,
                format_func=lambda p: str(
                    p.relative_to(GRAPHS)
                )
            )

            st.image(
                str(selected_graph),
                caption=selected_graph.name,
                use_container_width=True
            )

        else:
            st.warning("No research graphs found.")

    else:
        st.warning("Results graph folder not found.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Real-Time Intelligent Respiratory Monitoring and Analysis System | "
    "Research prototype | PhysioNet BIDMC dataset"
)
