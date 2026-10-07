# SomnoSync: Late-Night Screen Habit & Sleep Debt Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Classifier%20Accuracy-96.88%25-brightgreen.svg)](#model-1-sleep-debt-category-classifier)
[![R2 Fatigue](https://img.shields.io/badge/Fatigue%20R%C2%B2-0.943-success.svg)](#model-3-next-day-fatigue-score-regressor)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Mohammed-Salman-Hussain/Somno-Sync/pulls)

> **SomnoSync** is an end-to-end behavioral intelligence system and machine learning framework designed to quantify and predict the physiological cost of late-night digital device usage. Powered by an empirical dataset of 8,500 individuals, SomnoSync predicts **Sleep Debt Category**, **Sleep Latency (delay to fall asleep)**, and **Next-Day Fatigue Scores**, coupled with a responsive interactive dashboard and a lightweight Python inference backend.

---

## Table of Contents
- [Executive Overview](#executive-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [The Machine Learning Suite](#the-machine-learning-suite)
- [Empirical Dataset Overview](#empirical-dataset-overview)
- [Interactive Dashboard & UI](#interactive-dashboard--ui)
- [Inference REST API](#inference-rest-api)
- [Project Directory Structure](#project-directory-structure)
- [Quick Start & Setup](#quick-start--setup)
- [Running the Comprehensive Audit](#running-the-comprehensive-audit)
- [License & Author](#license--author)

---

## Executive Overview

### The Problem
In contemporary digital life, nighttime smartphone usage has become an ubiquitous routine. Users routinely spend 60–180 minutes consuming high-dopamine media (short-form video, social media, continuous streaming) immediately before attempting to sleep. This habit leads to:
1. **Prolonged Sleep Latency**: Exposure to blue light and cognitive stimulation inhibits natural melatonin secretion.
2. **Erosion of Sleep Architecture**: Reduced rapid eye movement (REM) and slow-wave restorative sleep.
3. **Cumulative Sleep Debt**: Chronic sleep deficit compounding across weekdays, leading to severe morning alarm snoozing, cognitive fatigue, and reduced daytime focus.

### The Solution
**SomnoSync** translates nighttime behavioral patterns into predictive intelligence:
- **Classifies Sleep Debt Severity** across 4 clinical tiers with **96.88% accuracy**.
- **Forecasts Exact Sleep Latency (Minutes)** before sleep onset ($R^2 = 0.846$, MAE 5.58 min).
- **Forecasts Next-Day Fatigue Level** on a 1.0–10.0 scale ($R^2 = 0.943$, MAE 0.48 pts).

---

## Key Features

- **Trained on 8,500 Validated Records**: Clean, zero-null dataset capturing 18 lifestyle, chronotype, and device metrics.
- **Production-Grade ML Pipeline**: Automated preprocessing using `ColumnTransformer` with `StandardScaler` for continuous metrics and `OneHotEncoder` for categoricals.
- **Trio of Specialized ML Models**: Serialized `.joblib` pipelines for multi-class classification and multi-variable regression.
- **Zero-Dependency Lightweight Backend**: Python built-in `http.server` implementation serving real-time predictions via `/api/predict`.
- **Interactive Nocturnal Dashboard**: Built with vanilla HTML5/CSS3/JavaScript, Chart.js, and HTML2Canvas for 1-click single-snapshot report generation.
- **Automated Audit Suite (`audit.py`)**: 5-stage automated test verifying data schema, model weights, metrics fidelity, live API responses, and assets.

---

## System Architecture

```
[Kaggle Dataset: 8,500 Records]
            │
            ▼
[Data Cleaning & Feature Engineering]
  ├── Categoricals: OneHotEncoder (Chronotype, Bedtime App, Occupation, Gender)
  └── Numericals: StandardScaler (Minutes, Brightness, Caffeine, Exercise, Age)
            │
            ├────────────────────────────────┬───────────────────────────────┐
            ▼                                ▼                               ▼
[Model 1: Debt Classifier]       [Model 2: Latency Regressor]     [Model 3: Fatigue Regressor]
Random Forest Classifier         Random Forest Regressor          Gradient Boosting Regressor
Predicts: sleep_debt_category    Predicts: sleep_latency_min      Predicts: next_day_fatigue_score
Accuracy: 96.88%                 R²: 0.8464 (MAE: 5.58 min)       R²: 0.9432 (MAE: 0.48 pts)
            │                                │                               │
            └────────────────────────────────┼───────────────────────────────┘
                                             ▼
                      [Evaluation Metrics & Serialized Artifacts]
                           (models/*.joblib & model_results.json)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [REST Inference Engine]                     [Interactive Dashboard]
         Python HTTP Server (server.py)                  index.html + Chart.js
         POST /api/predict (JSON)                        Live Simulator & Visualizer
```

---

## The Machine Learning Suite

### Model 1: Sleep Debt Category Classifier
- **Algorithm**: `RandomForestClassifier` (100 estimators)
- **Target**: `sleep_debt_category` (*Optimal Recovery*, *Mild Deficit*, *Moderate Debt*, *Severe Sleep Debt*)
- **Metrics**:
  - **Accuracy**: `96.88%`
  - **Macro F1-Score**: `0.935`
  - **Weighted F1-Score**: `0.967`
- **Dominant Features**: `total_sleep_hours` (60.4%), `morning_alarm_snoozes` (15.5%), `bedtime_phone_minutes` (6.7%), `chronotype` (7.7%).

### Model 2: Sleep Latency Regressor
- **Algorithm**: `RandomForestRegressor`
- **Target**: `sleep_latency_min` (Continuous minutes required to fall asleep)
- **Metrics**:
  - **$R^2$ Score**: `0.8464`
  - **MAE**: `5.58 minutes`
  - **RMSE**: `7.04 minutes`
- **Dominant Features**: `bedtime_phone_minutes` (81.6%), `caffeine_post_5pm_mg` (5.3%), `primary_bedtime_app` (TikTok/Reels: 4.5%), `screen_brightness_pct` (2.8%).

### Model 3: Next-Day Fatigue Score Regressor
- **Algorithm**: `GradientBoostingRegressor`
- **Target**: `next_day_fatigue_score` (Continuous scale 1.0 to 10.0)
- **Metrics**:
  - **$R^2$ Score**: `0.9432`
  - **MAE**: `0.48 points`
  - **RMSE**: `0.65 points`
- **Dominant Features**: `morning_alarm_snoozes` (51.3%), `total_sleep_hours` (40.2%), `bedtime_phone_minutes` (4.5%).

---

## Empirical Dataset Overview

The dataset (`bedtime_screentime_sleep_debt.csv`) encompasses **8,500 complete observations** across 18 distinct variables:

| Feature Category | Features Included |
| :--- | :--- |
| **Demographics** | `age`, `gender`, `occupation_type` |
| **Circadian Profile** | `chronotype` (*Morning Lark*, *Night Owl*, *Intermediate*) |
| **Digital Habits** | `bedtime_phone_minutes`, `primary_bedtime_app`, `screen_brightness_pct`, `blue_light_filter_active` |
| **Physiological/Lifestyle** | `caffeine_post_5pm_mg`, `physical_activity_min`, `total_sleep_hours`, `morning_alarm_snoozes` |
| **Target Outputs** | `sleep_debt_category`, `sleep_latency_min`, `next_day_fatigue_score` |

---

## Interactive Dashboard & UI

The frontend is contained in `index.html`, designed specifically with a sleek **Nocturnal Dark Theme** (deep slate, violet accents, crisp typography) with no external framework dependencies:
- **Executive KPI Cards**: Real-time high-level figures across all 8,500 records.
- **Model Benchmark Displays**: Clear performance badges, confusion matrix breakdown, and feature importance bar graphs.
- **Live Habit Simulator**: Sliders and selectors to test hypothetical scenarios (e.g., Doomscrolling vs. Reading) with instantaneous feedback.
- **Data Explorer Table**: Sortable preview of raw records with search filtering.
- **Single-Click Snapshot Export**: Integrated `html2canvas` module allowing instant high-definition export of the entire dashboard for reports and presentations.

---

## Inference REST API

The included `server.py` hosts a lightweight, CORS-enabled HTTP service for serving predictions from the serialized `.joblib` models.

### Endpoint: `POST /api/predict`

#### Request Payload
```json
{
  "age": 24,
  "gender": "Female",
  "occupation_type": "Corporate 9-to-5",
  "chronotype": "Night Owl",
  "bedtime_phone_minutes": 120,
  "primary_bedtime_app": "TikTok / Reels",
  "screen_brightness_pct": 80,
  "blue_light_filter_active": 0,
  "caffeine_post_5pm_mg": 80,
  "physical_activity_min": 15,
  "total_sleep_hours": 5.0,
  "morning_alarm_snoozes": 4
}
```

#### Response
```json
{
  "status": "success",
  "source": "Python Scikit-Learn Production Models",
  "predictions": {
    "model_1_sleep_debt_category": "Severe Sleep Debt",
    "model_1_class_probabilities": {
      "Mild Deficit": 0.0,
      "Moderate Debt": 11.2,
      "Optimal Recovery": 0.0,
      "Severe Sleep Debt": 88.8
    },
    "model_2_sleep_latency_min": 78.4,
    "model_3_next_day_fatigue_score": 7.9
  }
}
```

---

## Project Directory Structure

```text
Somno-Sync/
├── charts/                                  # High-resolution analytical plots
│   ├── latency_by_app.png
│   └── latency_vs_screentime.png
├── models/                                  # Trained & serialized model pipelines
│   ├── fatigue_regressor.joblib
│   ├── sleep_debt_classifier.joblib
│   └── sleep_latency_regressor.joblib
├── .gitignore                               # Standard git ignore definitions
├── audit.py                                 # Automated 5-stage project audit script
├── bedtime_screentime_sleep_debt.csv        # Empirical dataset (8,500 rows)
├── dashboard_data.json                      # Aggregated dataset metrics for UI
├── index.html                               # Interactive single-page dashboard
├── LICENSE                                  # MIT open-source license
├── load_dataset.py                          # Kagglehub dataset ingestion script
├── model_results.json                       # Comprehensive evaluation metrics & confusion matrix
├── project_ideas.md                         # Catalog of 6 data science & ML research ideas
├── README.md                                # Project documentation & guide
├── REPORT.md                                # In-depth technical architecture report
├── requirements.txt                         # Python dependencies
├── server.py                                # Lightweight HTTP inference backend
└── train_models.py                          # Full training pipeline script
```

---

## Quick Start & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Mohammed-Salman-Hussain/Somno-Sync.git
cd Somno-Sync
```

### 2. Set Up Python Environment
```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Re-Train the Models
All models are pre-trained and serialized in `./models/`. To execute a clean training run from scratch:
```bash
python train_models.py
```

### 4. Start the Inference Server & Dashboard
```bash
python server.py
```
Open your browser and navigate to:
```text
http://localhost:8000/index.html
```
*(Alternatively, you can open `index.html` directly in any modern browser for client-side demo mode).*

---

## Running the Comprehensive Audit

To verify dataset integrity, model weights, metrics alignment, and live inference:
1. Ensure the server is running in a terminal:
   ```bash
   python server.py
   ```
2. In a separate terminal, run:
   ```bash
   python audit.py
   ```
Output:
```text
==================================================
      SOMNOSYNC COMPREHENSIVE PROJECT AUDIT
==================================================

[1/5] Auditing Dataset...
  [OK] Dataset verified: 8500 records, 18 features, 0 null values.

[2/5] Auditing Model Artifacts in ./models/ ...
  [OK] models/sleep_debt_classifier.joblib loaded successfully: RandomForestClassifier
  [OK] models/sleep_latency_regressor.joblib loaded successfully: RandomForestRegressor
  [OK] models/fatigue_regressor.joblib loaded successfully: GradientBoostingRegressor

[3/5] Auditing Model Evaluation Metrics in model_results.json ...
  [OK] Model 1 (Random Forest Classifier) Accuracy: 96.88% (Macro F1: 0.935)
  [OK] Model 2 (Random Forest Regressor) R2 Score:  0.8464 (MAE: 5.58 min)
  [OK] Model 3 (Gradient Boosting Regressor) R2:    0.9432 (MAE: 0.48 pts)

[4/5] Auditing Live Backend Server (http://localhost:8000)...
  [OK] Healthy Sleep Habits (15m screen, 8h sleep)
  [OK] Heavy Doomscrolling (140m screen, 4.5h sleep)

[5/5] Auditing Project Deliverables & Files...
  [OK] REPORT.md
  [OK] project_ideas.md
  [OK] index.html
  [OK] train_models.py
  ...
==================================================
      ALL AUDIT CHECKS PASSED WITH 100% SUCCESS!
==================================================
```

---

## License & Author

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

**Author**: **Mohammed Salman Hussain**  
- **GitHub**: [@Mohammed-Salman-Hussain](https://github.com/Mohammed-Salman-Hussain)  
- **Email**: [salmanhussain1199@gmail.com](mailto:salmanhussain1199@gmail.com)  
- **Repository**: [https://github.com/Mohammed-Salman-Hussain/Somno-Sync](https://github.com/Mohammed-Salman-Hussain/Somno-Sync)
