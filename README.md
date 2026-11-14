# Intelligent Respiratory Monitoring and Analysis System

A Python-based biomedical signal-processing project for respiratory waveform analysis, automatic breath detection, respiratory-rate estimation, model evaluation, and signal-quality investigation.

**Live dashboard:** https://respiratory-monitoring-system.streamlit.app/  
**Dataset:** [PhysioNet BIDMC PPG and Respiration Dataset](https://physionet.org/content/bidmc/1.0.0/)

> **Scope:** This is an engineering and research prototype. It analyzes recorded signals and is not a clinically validated medical device.

## Project overview

The system processes respiratory recordings from the BIDMC dataset and compares conventional signal-processing methods with machine-learning regressors. A Streamlit dashboard provides interactive access to selected recordings, analysis outputs, evaluation reports, and research visualizations.

The project is organized around this pipeline:

```text
BIDMC recordings + reference annotations
                 |
                 v
       Signal preprocessing
                 |
                 v
       Breath-event detection
                 |
                 v
      Respiratory-rate estimates
                 |
       +---------+----------+
       |                    |
       v                    v
 Feature extraction   Signal-quality audit
       |
       v
 Grouped model evaluation
       |
       v
 Reports, plots and dashboard
```

## What is implemented

- Butterworth band-pass filtering of respiratory waveforms.
- Automatic breath-event detection using peak-based methods.
- Respiratory-rate estimation from detected events.
- Comparison against reference breath annotations.
- Time-domain and frequency-domain feature extraction.
- Evaluation of Random Forest, Extra Trees, and Gradient Boosting regressors.
- Recording-wise data separation and grouped cross-validation to reduce recording-level leakage.
- Signal-quality and annotation-disagreement investigations.
- Interactive Streamlit dashboard with Plotly visualizations.

## Research approach and findings

The machine-learning models are evaluated against a conventional signal-processing baseline rather than assumed to be better.

The committed evaluation reports show that the signal-processing baseline performed strongly in grouped cross-validation. Model rankings vary between evaluation protocols, so results should be interpreted with the split strategy and dataset limitations in mind.

For reproducibility, consult the CSV reports in `results/reports/` alongside the analysis scripts. Do not compare metrics from different splits as if they came from the same test set.

## Evaluation documentation

- [Model evaluation notes](docs/evaluation-notes.md) — grouped cross-validation and pooled out-of-fold metrics, interpretation, and reproducibility checklist.

## Dashboard

The hosted Streamlit app provides sections for:

- Project overview
- Respiratory signal exploration
- Respiratory analysis
- Model evaluation
- Signal-quality investigation
- Research graphs

The public deployment includes two selected recordings to keep the hosted application lightweight. The complete BIDMC dataset is not bundled with this repository.

## Dataset

The project uses the [BIDMC PPG and Respiration Dataset, version 1.0.0](https://physionet.org/content/bidmc/1.0.0/), which contains 53 recordings with physiological signals and reference annotations.

Download the dataset from PhysioNet and place the extracted CSV data in the directory expected by the processing scripts. The feature-extraction script currently expects:

```text
data/raw/BIDMC/bidmc-1.0.0/bidmc_csv/
```

The dashboard reads the selected deployment recordings from:

```text
data/deployment/
```

## Repository structure

```text
respiratory-monitoring-system/
├── dashboard/
│   └── app.py
├── data/
│   └── deployment/          # Selected sample recordings for the dashboard
├── results/
│   ├── graphs/
│   ├── models/
│   └── reports/
├── src/
│   ├── analysis/
│   ├── processing/
│   ├── visualization/
│   ├── feature_extraction.py
│   ├── prepare_ml_dataset.py
│   ├── train_models.py
│   └── signal_quality_audit.py
├── requirements.txt
└── README.md
```

## Run locally

### Requirements

- Python 3.10 or newer (Python 3.12 is used for development)
- Git
- Internet access to obtain the original dataset if you want to reproduce the full analysis

### 1. Clone the repository

```bash
git clone https://github.com/challabhavithsai0123-cell/respiratory-monitoring-system.git
cd respiratory-monitoring-system
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit.

## Reproducing the analysis

The dashboard can run with the selected deployment recordings and committed reports. Reproducing the complete dataset analysis requires downloading the full BIDMC dataset and placing it in the expected data directory.

The main analysis scripts include:

- `src/feature_extraction.py` — extracts respiratory features from the full dataset.
- `src/prepare_ml_dataset.py` — prepares the supervised-learning dataset and records annotation-quality labels.
- `src/train_models.py` — trains and evaluates the regression models.

Run research scripts from the repository root so their relative input and output paths resolve correctly. Check each script's configuration before running it against a new dataset layout.

## Limitations and future work

Current limitations:

- The current workflow analyzes recorded data; it does not acquire live patient signals from a wearable sensor.
- The hosted dashboard contains only selected sample recordings.
- Model performance and generalization are limited by the size and characteristics of the BIDMC dataset.
- The system has not undergone clinical validation.

Potential extensions:

- Real-time respiratory-sensor acquisition.
- Embedded signal acquisition and edge processing.
- More robust artifact rejection and signal-quality estimation.
- Evaluation on additional datasets and subject-independent protocols.
- Live streaming and monitoring interfaces.

## Disclaimer

This repository is intended for educational, engineering, and research use. It is not intended for diagnosis, treatment decisions, or clinical patient monitoring.

## Author

**C. Bhavith Sai**  
B.Tech — Mechatronics Engineering  
[GitHub profile](https://github.com/challabhavithsai0123-cell)

---

**Project areas:** Biomedical Signal Processing · Respiratory Analysis · Python · SciPy · Machine Learning · Streamlit


## Engineering portfolio website

The responsive portfolio site is maintained in the [`docs/`](docs/) directory. To publish it free with GitHub Pages, open **Settings → Pages**, choose **Deploy from a branch**, select `main` and `/docs`, then save. GitHub will show the published URL in that section once deployment completes.
