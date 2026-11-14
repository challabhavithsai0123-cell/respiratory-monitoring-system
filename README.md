INTELLIGENT RESPIRATORY MONITORING AND ANALYSIS SYSTEM

A Python-based biomedical signal processing and analysis system for respiratory waveform filtering, automatic breath detection, respiratory rate estimation, machine learning evaluation, and signal-quality assessment.

Live Dashboard: [https://respiratory-monitoring-system.streamlit.app/](https://respiratory-monitoring-system.streamlit.app/)
GitHub Repository: [https://github.com/challabhavithsai0123-cell/respiratory-monitoring-system](https://github.com/challabhavithsai0123-cell/respiratory-monitoring-system)

1. Project Overview
The Intelligent Respiratory Monitoring and Analysis System is a software-based biomedical engineering project designed to process and analyze respiratory signals. The system uses recorded physiological data from the PhysioNet BIDMC PPG and Respiration Dataset to investigate respiratory waveform characteristics, detect breathing events, estimate respiratory rates, and evaluate different estimation approaches. A Streamlit dashboard provides interactive visualization of respiratory signals, analysis results, model evaluation metrics, and signal-quality investigations. The project focuses on respiratory signal processing, algorithm development, data-driven analysis, and research-oriented visualization.
2. Key Features

* Respiratory signal filtering using a Butterworth bandpass filter.
* Automatic breath detection using peak detection techniques.
* Respiratory rate estimation from detected breathing events.
* Respiratory feature extraction using time-domain and frequency-domain methods.
* Machine learning model development and evaluation.
* Recording-wise and cross-validation performance analysis.
* Signal-quality and artifact investigation.
* Interactive respiratory waveform visualization.
* Web-based dashboard using Streamlit and Plotly.

3. System Architecture
The project follows a modular processing pipeline:
4. Dataset Acquisition: Load respiratory recordings and reference breath annotations from the PhysioNet BIDMC dataset.
5. Signal Processing: Apply digital filtering to extract relevant respiratory waveform characteristics.
6. Breath Detection: Identify candidate breathing events from the filtered signal.
7. Feature Extraction: Calculate statistical, temporal, and spectral features.
8. Respiratory Rate Estimation: Estimate breathing rate using detected events.
9. Machine Learning Evaluation: Compare Random Forest, Extra Trees, and Gradient Boosting models against the baseline signal-processing approach.
10. Signal Quality Analysis: Investigate annotation disagreements, signal artifacts, and recording-level performance.
11. Visualization: Present the results through an interactive Streamlit dashboard.
12. Technologies Used

* Programming Language: Python 3.12
* Signal Processing: SciPy
* Numerical Computing: NumPy
* Data Analysis: Pandas
* Visualization: Matplotlib, Plotly
* Dashboard: Streamlit
* Machine Learning: Scikit-learn
* Model Storage: Joblib
* Dataset: PhysioNet BIDMC

5. Machine Learning Models
The project evaluates three machine learning approaches:

* Random Forest Regressor
* Extra Trees Regressor
* Gradient Boosting Regressor

Their performance is compared with a conventional signal-processing baseline. The evaluation uses recording-wise data separation to reduce data leakage between training and testing recordings.

Important research observation: The baseline signal-processing approach performed better on average during grouped cross-validation. Extra Trees achieved a slightly lower MAE than the baseline on the held-out test recordings. Therefore, machine learning is treated as an experimental comparison rather than a universally superior estimator.

6. Interactive Dashboard
The deployed Streamlit application contains six sections:
7. Overview: Project summary and system architecture.
8. Signal Explorer: Interactive respiratory waveform exploration.
9. Respiratory Analysis: Breath detection and respiratory rate analysis.
10. Model Evaluation: Comparison of machine learning models and baseline performance.
11. Signal Quality: Investigation of artifacts and signal-quality characteristics.
12. Research Graphs: Visualization of experimental findings and evaluation results.

Live Application: [https://respiratory-monitoring-system.streamlit.app/](https://respiratory-monitoring-system.streamlit.app/)

7. Dataset
PhysioNet BIDMC PPG and Respiration Dataset
Source: [https://physionet.org/content/bidmc/1.0.0/](https://physionet.org/content/bidmc/1.0.0/)

The original research uses 53 recordings containing respiratory and physiological signals with reference annotations. The deployed dashboard includes two selected respiratory signal recordings to keep the hosted application lightweight. The complete original dataset is not included in this GitHub repository.

8. Project Structure
Respiratory_Monitoring_System/
├── dashboard/
│   └── app.py
├── data/
│   └── deployment/
│       ├── bidmc_01_Signals.csv
│       └── bidmc_13_Signals.csv
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
├── README.md
└── .gitignore
9. Installation and Execution
Prerequisites:

* Python 3.12
* Git
* Internet connection

Clone the repository:
git clone [https://github.com/challabhavithsai0123-cell/respiratory-monitoring-system.git](https://github.com/challabhavithsai0123-cell/respiratory-monitoring-system.git)
cd respiratory-monitoring-system

Create a virtual environment:
python -m venv .venv

Activate the environment (Windows PowerShell):
..venv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

Run the dashboard:
python -m streamlit run dashboard/app.py

Open the local URL displayed in the terminal to access the dashboard.

Note: The deployment folder contains only two sample recordings. Reproducing the complete original research analysis requires obtaining the full BIDMC dataset and placing it in the expected local data directory.

10. Research and Evaluation
The project investigates:

* Respiratory waveform filtering and breath detection.
* Agreement between automatic detections and reference annotations.
* Respiratory rate estimation accuracy.
* Recording-wise generalization of machine learning models.
* Effects of signal artifacts and annotation disagreements.
* Differences between conventional signal processing and machine learning approaches.

The results are intended for engineering and research evaluation.

11. Limitations and Future Scope
Current limitations:

* The system analyzes previously recorded physiological signals rather than acquiring live patient data.
* Only two selected recordings are included in the public dashboard deployment.
* The machine learning models have limitations in generalizing to unseen recordings.
* The system has not undergone clinical validation.

Potential future improvements:

* Integration with real-time respiratory sensor acquisition.
* Embedded hardware integration for physiological signal monitoring.
* Improved artifact rejection and signal-quality assessment.
* Evaluation on larger and more diverse datasets.
* Real-time streaming and visualization.

12. Important Disclaimer
This project is an engineering and research prototype intended for educational and experimental purposes. It is not a clinically validated medical device and must not be used for medical diagnosis, treatment decisions, or patient monitoring.

Project Category: Biomedical Signal Processing | Respiratory Analysis | Machine Learning | Python | Streamlit
