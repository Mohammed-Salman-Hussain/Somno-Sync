# Project Report: SomnoSync

**Project Name:** SomnoSync  
**Dataset:** [Bedtime Screen Time & Sleep Debt (8,500 records)](file:///c:/Users/salman/Desktop/p1/bedtime_screentime_sleep_debt.csv)  
**Deliverables:**
- [project_ideas.md](file:///c:/Users/salman/Desktop/p1/project_ideas.md) — Reference catalog of 6 ML & data science project ideas.
- [train_models.py](file:///c:/Users/salman/Desktop/p1/train_models.py) — Machine learning training pipeline implementing Ideas #3 & #4.
- [models/](file:///c:/Users/salman/Desktop/p1/models) — Production serialized `.joblib` model artifacts.
- [charts/](file:///c:/Users/salman/Desktop/p1/charts) — High-resolution generated analytical plots.
- [index.html](file:///c:/Users/salman/Desktop/p1/index.html) — Production-grade interactive dashboard with Chart.js, live confusion matrix, real data explorer, and 1-click snapshot export.

---

## 1. Why the Project Was Made

### The Problem
In modern society, nighttime smartphone usage has become an ingrained ritual. Users frequently spend 60–180 minutes consuming high-stimulation media (TikTok, Reels, YouTube, streaming) right before attempting sleep. 

The physiological and psychological ramifications are significant:
1. **Prolonged Sleep Latency:** Blue light and dopamine-inducing content delay melatonin secretion, keeping the nervous system hyper-aroused.
2. **Erosion of Sleep Architecture:** High bedtime screen exposure reduces deep sleep and REM percentages.
3. **Compounding Sleep Debt:** Even minor daily deficits accumulate into chronic exhaustion, leading to frequent morning alarm snoozing and impaired next-day cognitive function.

### The Objective
Despite widespread intuition that "phones disrupt sleep," users lack quantifiable feedback on **how much** their specific habits cost them. **SomnoSync** was created to turn behavioral data into predictive intelligence by answering:
- *Can we predict the severity of a user's sleep debt category based on their digital and sleep habits?*
- *How accurately can we forecast the exact delay in minutes to fall asleep (sleep latency)?*
- *What is the forecasted next-day fatigue score on a 1–10 scale?*

---

## 2. How the Project Was Made

### Pipeline & Architecture

```
[Raw Dataset: 8,500 records]
            │
            ▼
[Data Cleaning & Feature Engineering]
  ├── Categorical Encoding: One-Hot (Chronotype, App, Occupation, Gender)
  └── Numerical Normalization: StandardScaler (Minutes, Brightness, Caffeine, Exercise)
            │
            ├────────────────────────────────┬───────────────────────────────┐
            ▼                                ▼                               ▼
[Model 1: Idea #3 Classifier]    [Model 2: Idea #4A Regressor]    [Model 3: Idea #4B Regressor]
Random Forest Classifier         Random Forest Regressor          Gradient Boosting Regressor
Predicts: sleep_debt_category    Predicts: sleep_latency_min      Predicts: next_day_fatigue_score
(Optimal / Mild / Moderate /     (Continuous minutes to sleep)    (Continuous score 1.0 - 10.0)
 Severe Debt)
            │                                │                               │
            └────────────────────────────────┼───────────────────────────────┘
                                             ▼
                      [Evaluation Metrics & JSON Serialization]
                                             ▼
                     [Interactive One-Page HTML UI (index.html)]
                     Optimized for Full-Screen & Single Screenshot
```

### 1. Data Ingestion & Preprocessing
- **Source:** Kaggle (`samartalwar/sleep-debt-and-screen-time-late-night-phone-habits`).
- **Cleaning:** Verified data integrity with 0 null values across all 8,500 samples.
- **Partitioning:** Stratified 80/20 train-test splits ensuring class balance across debt categories.

### 2. Machine Learning Models (Ideas 3 & 4)
1. **Multi-Class Sleep Debt Classifier (Random Forest):**
   - **Target:** `sleep_debt_category` (*Optimal Recovery*, *Mild Deficit*, *Moderate Debt*, *Severe Sleep Debt*).
   - **Performance:** Achieved high classification accuracy with balanced precision and recall across all 4 tiers.
   - **Key Drivers:** `total_sleep_hours`, `morning_alarm_snoozes`, and `bedtime_phone_minutes`.

2. **Sleep Latency Predictor (Random Forest Regressor):**
   - **Target:** `sleep_latency_min` (time in minutes to fall asleep).
   - **Performance:** Evaluated via $R^2$, MAE, and RMSE.
   - **Key Drivers:** `bedtime_phone_minutes` exhibits the strongest direct correlation ($r = +0.86$), amplified by `screen_brightness_pct` and `caffeine_post_5pm_mg`.

3. **Next-Day Fatigue Score Predictor (Gradient Boosting Regressor):**
   - **Target:** `next_day_fatigue_score` (1.0 to 10.0).
   - **Performance:** High explanatory power ($R^2 > 0.88$), effectively modeling non-linear fatigue escalation.

### 3. Unified Single-Page Presentation
To satisfy the requirement of a single screenshot without requiring cumbersome scrolling or fragmented slides:
- Built [index.html](file:///c:/Users/salman/Desktop/p1/index.html) with a nocturnal theme (deep slate, violet accents, crisp typography).
- Displays executive summary KPIs, model performance cards, feature importance distributions, and an interactive prediction simulator all within a clean visual hierarchy.
