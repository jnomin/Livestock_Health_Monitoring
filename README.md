# 🐄 Livestock Health Monitoring Digital Twin System

A precision livestock farming system that integrates IoT sensor data, machine learning, and digital twin technology to enable early detection of behavioral anomalies in cattle.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Digital Twin — CowTwin](#digital-twin--cowtwin)
- [Dashboard](#dashboard)
- [Results](#results)
- [Project Structure](#project-structure)

---

## Overview

This project presents the design and implementation of a **Livestock Health Monitoring Digital Twin** system. It processes triaxial accelerometer collar data from cattle, classifies behavioral states using machine learning, detects anomalies, and maintains a virtual health representation (digital twin) per animal — providing herders with an automated behavioral monitoring and early warning system.

**Key capabilities:**
- Classifies cattle behavior into Eating, Ruminating, and Other from raw sensor data
- Detects anomalous behavioral windows that deviate from learned normal patterns
- Maintains per-animal NORMAL / AT_RISK / ALERT health state transitions
- Visualizes behavior timelines and alerts through an interactive dashboard

---

## Dataset

**UK Precision Beef Dataset**
- **Source:** [Zenodo — DOI: 10.5281/zenodo.4064801](https://doi.org/10.5281/zenodo.4064801)
- **Location:** Easter Howgate Farm, Edinburgh, UK
- **Animals:** 18 steers across 3 farm trials
- **Sensor:** Triaxial accelerometer collar
- **Sampling rate:** 10 Hz (10 readings per second)
- **Behavior labels:** Eating, Ruminating, Other

Download the dataset from the Zenodo link above and place the raw CSV files into the `data/raw/` directory before running the pipeline.

---

## System Architecture

```
Physical Sensors (10Hz collar data)
          │
          ▼
  Data Ingestion & Storage
          │
          ▼
  Preprocessing Pipeline
  (segmentation → feature extraction → normalization)
          │
          ▼
  ┌───────────────────────────┐
  │   Machine Learning Layer  │
  │  ┌─────────────────────┐  │
  │  │  Random Forest       │  │   ← Behavior classification
  │  │  Classifier          │  │
  │  └─────────────────────┘  │
  │  ┌─────────────────────┐  │
  │  │  Isolation Forest    │  │   ← Anomaly detection
  │  │  Anomaly Detector    │  │
  │  └─────────────────────┘  │
  └───────────────────────────┘
          │
          ▼
  CowTwin — Digital Twin Layer
  (per-animal state: NORMAL → AT_RISK → ALERT)
          │
          ▼
  Visualization Dashboard
  (Streamlit / Dash — behavior timeline, alerts)
```

---

## Features

- **Window segmentation** — raw 10 Hz data segmented into 30-second windows (300 samples/window/axis)
- **19-feature extraction** — signal magnitude, ODBA, per-axis mean/std/min/max/skewness, and more
- **Cross-animal validation** — train on 2 animals, evaluate on a completely unseen 3rd animal
- **Behavioral classification** — Random Forest predicting Eating, Ruminating, or Other per window
- **Anomaly detection** — Isolation Forest flagging windows that deviate from normal behavioral patterns
- **Digital twin state machine** — CowTwin class tracking consecutive anomaly streaks and transitioning health states
- **Interactive dashboard** — real-time-style behavior timeline and alert panel per animal

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/your-username/livestock-digital-twin.git
cd livestock-digital-twin

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit          # or dash
joblib
scipy
```

---

## Usage

Run the full pipeline end-to-end:

```bash
python main.py
```

Or run each stage individually:

```bash
# 1. Preprocess raw sensor data
python src/preprocessing.py

# 2. Train and evaluate the Random Forest classifier
python src/train_classifier.py

# 3. Run the Isolation Forest anomaly detector
python src/anomaly_detection.py

# 4. Simulate the CowTwin digital twin
python src/digital_twin.py

# 5. Launch the dashboard
streamlit run src/dashboard.py
```

---

## Data Preprocessing

Raw accelerometer readings (X, Y, Z axes at 10 Hz) are processed through the following pipeline:

**1. Window Segmentation**
Raw time-series data is divided into non-overlapping 30-second windows (300 samples per window per axis). Each window is assigned a behavior label via majority vote of the per-sample annotations.

**2. Feature Extraction**
19 statistical features are extracted per window:

| Feature Group | Features |
|---|---|
| Per-axis statistics | Mean, standard deviation, min, max, skewness (×3 axes = 15 features) |
| Signal Magnitude Area (SMA) | Combined XYZ magnitude proxy |
| Overall Dynamic Body Acceleration (ODBA) | Sum of absolute deviations from mean per axis |
| Signal Magnitude (resultant) | √(X² + Y² + Z²) mean |

**3. Normalization**
Features are standardized using `StandardScaler` fitted on the training set only, then applied to the test set to prevent data leakage.

**4. Outlier Removal**
IQR-based outlier detection is applied per feature. Flagged windows are removed before model training.

Output: a clean feature matrix saved to `data/processed/features.csv` and `data/processed/labels.csv`.

---

## Machine Learning Models

### Random Forest Classifier

Classifies each 30-second window into one of three behavioral states:

- **Eating**
- **Ruminating**
- **Other**

**Training strategy:** leave-one-animal-out cross-validation. The model is trained on animals 1 and 2, then evaluated on a completely unseen animal 3 to test cross-animal generalizability.

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

**Achieved accuracy:** 75% on the unseen test animal.

### Isolation Forest Anomaly Detector

Trained on windows from the test animal's "normal" periods. Flags windows that deviate significantly from the learned behavioral distribution — without requiring labeled anomaly examples.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.10, random_state=42)
iso.fit(X_normal)
anomaly_scores = iso.decision_function(X_test_windows)
anomaly_labels  = iso.predict(X_test_windows)   # -1 = anomaly, 1 = normal
```

**Key result:** 14.7% of test animal windows flagged as anomalous; maximum sustained anomaly streak of 37 consecutive windows (18 continuous minutes of unusual behavior detected).

---

## Digital Twin — CowTwin

The `CowTwin` class maintains a persistent virtual health state for each individual animal. It consumes the per-window output from the ML layer and applies threshold-based state transitions.

### Health States

| State | Meaning |
|---|---|
| `NORMAL` | Behavior within expected range; no action required |
| `AT_RISK` | Elevated anomaly streak detected; monitor closely |
| `ALERT` | Sustained anomalous behavior; intervention recommended |

### State Transition Logic

```python
class CowTwin:
    def __init__(self, animal_id, at_risk_threshold=5, alert_threshold=15):
        self.animal_id = animal_id
        self.state = "NORMAL"
        self.anomaly_streak = 0
        self.at_risk_threshold = at_risk_threshold
        self.alert_threshold = alert_threshold

    def update(self, is_anomaly: bool):
        if is_anomaly:
            self.anomaly_streak += 1
        else:
            self.anomaly_streak = 0

        if self.anomaly_streak >= self.alert_threshold:
            self.state = "ALERT"
        elif self.anomaly_streak >= self.at_risk_threshold:
            self.state = "AT_RISK"
        else:
            self.state = "NORMAL"

        return self.state
```

Each animal in the herd gets its own `CowTwin` instance. The twin is updated once per 30-second window, maintaining a full history of state transitions for trend analysis.

---

## Dashboard

The dashboard provides a real-time-style monitoring interface for herd behavioral health.

**Launch:**

```bash
streamlit run src/dashboard.py
```

**Features:**
- Per-animal behavior timeline (Eating / Ruminating / Other) across the monitoring period
- Anomaly overlay — flagged windows highlighted on the timeline
- Health state indicator per animal (NORMAL / AT_RISK / ALERT) with streak count
- Herd summary panel showing aggregate anomaly rates
- Alert log listing sustained anomaly events with timestamps and duration

---

## Results

| Metric | Value |
|---|---|
| Behavior classification accuracy (cross-animal) | 75% |
| Training animals | 2 (animals 1 & 2) |
| Test animal | 1 (completely unseen) |
| Behavior classes | Eating, Ruminating, Other |
| Anomaly rate (test animal) | 14.7% of windows |
| Maximum anomaly streak | 37 consecutive windows |
| Maximum continuous anomaly duration | 18 minutes |

---

## Project Structure

```
livestock-digital-twin/
│
├── data/
│   ├── raw/                  # Raw CSV files from Zenodo dataset
│   └── processed/            # Feature matrices and labels after preprocessing
│
├── models/
│   ├── random_forest.pkl     # Saved trained classifier
│   └── isolation_forest.pkl  # Saved anomaly detector
│
├── src/
│   ├── preprocessing.py      # Window segmentation and feature extraction
│   ├── train_classifier.py   # Random Forest training and evaluation
│   ├── anomaly_detection.py  # Isolation Forest training and scoring
│   ├── digital_twin.py       # CowTwin class and simulation runner
│   └── dashboard.py          # Streamlit dashboard
│
├── notebooks/
│   └── exploration.ipynb     # EDA and prototyping
│
├── main.py                   # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Citation

If you use this project or the dataset, please cite:

```
UK Precision Beef Dataset
Zenodo DOI: 10.5281/zenodo.4064801
Easter Howgate Farm, Edinburgh, UK
```
