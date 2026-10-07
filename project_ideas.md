# Project Ideas: Sleep Debt & Late-Night Screen Time Dataset

This document archives the 6 project ideas based on the dataset [`samartalwar/sleep-debt-and-screen-time-late-night-phone-habits`](file:///c:/Users/salman/Desktop/p1/bedtime_screentime_sleep_debt.csv).

---

## 1. Exploratory Data Analysis (EDA) & Sleep Insights Report (Beginner)
- **Goal:** Uncover how screen time, specific apps, and evening lifestyle factors influence sleep latency and sleep quality.
- **Key Analysis Areas:**
  - Short-form video apps (*TikTok/Reels*) vs. reading apps (*News/Books*) impact on sleep latency.
  - Effectiveness of the blue light filter across various screen brightness levels.
  - Interaction between evening caffeine intake (post-5pm) and sleep latency.
  - Deep sleep vs. REM sleep proportions across different chronotypes (Morning Lark, Intermediate, Night Owl).
- **Tools/Tech:** Python, Pandas, Matplotlib, Seaborn, Jupyter Notebook.

---

## 2. Interactive Streamlit / Dash Health Dashboard (Beginner / Intermediate)
- **Goal:** Provide a user-facing interactive portal where individuals can filter data by demographics and habits to benchmark their sleep health.
- **Key Features:**
  - Dynamic filters for occupation, chronotype, age bracket, and bedtime app.
  - Interactive distribution plots (sleep latency, total sleep hours, fatigue score).
  - Habit comparison tool: compare "your nightly profile" against the broader population.
- **Tools/Tech:** Streamlit / Plotly Dash, Altair / Plotly.

---

## 3. Sleep Debt Category Classifier (Multi-class Classification)
- **Goal:** Predict a user's sleep debt severity category (`Optimal Recovery`, `Mild Deficit`, `Moderate Debt`, `Severe Sleep Debt`) using their bedtime phone habits and daytime routines.
- **Key Features:**
  - Feature engineering: encoding categorical variables (chronotype, bedtime app, occupation) and scaling continuous metrics.
  - Model benchmarking: Logistic Regression, Random Forest Classifier, Gradient Boosting (XGBoost / LightGBM).
  - Model interpretability: Feature importance analysis and confusion matrix evaluation.
- **Tools/Tech:** Scikit-Learn, XGBoost, Joblib.

---

## 4. Fatigue & Sleep Latency Predictor (Supervised Regression)
- **Goal:** Quantitatively predict continuous sleep outcomes:
  - `sleep_latency_min`: Estimated minutes required to fall asleep.
  - `next_day_fatigue_score`: Rated on a scale from 1.0 to 10.0.
- **Key Features:**
  - Evaluating linear vs. non-linear regressors (Ridge/Lasso, Random Forest Regressor, Gradient Boosted Trees).
  - Performance metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.
  - Residual diagnostics to identify non-linear saturation effects of high screen time.
- **Tools/Tech:** Scikit-Learn, NumPy, Matplotlib.

---

## 5. Nighttime Behavioral Clustering & User Personas (Unsupervised Learning)
- **Goal:** Discover natural clusters of late-night digital behaviors without predefined labels.
- **Key Features:**
  - Dimensionality reduction with PCA and t-SNE / UMAP for 2D visualization.
  - Clustering with K-Means, Hierarchical Clustering, or Gaussian Mixture Models (GMM).
  - Persona characterization: e.g., *"Late-Night Doomscrollers"*, *"Caffeine-Fueled Shift Workers"*, *"Mindful Sleepers"*.
- **Tools/Tech:** Scikit-Learn, SciPy, UMAP / Seaborn.

---

## 6. Smart Sleep Coach & Intervention Recommender System (Advanced)
- **Goal:** Build a counterfactual intervention simulator that prescribes actionable habit changes to reduce fatigue and sleep debt.
- **Key Features:**
  - What-If Simulator: e.g., *"If you drop bedtime phone time from 90 min to 30 min and enable blue light filter, expected sleep latency drops by 32 min and fatigue score drops by 2.4."*
  - Sensitivity analysis per user demographic profile.
  - Prescription engine that prioritizes the highest ROI behavioral intervention.
- **Tools/Tech:** Python, Causal Inference / Predictive Regressors, FastAPI or Streamlit UI.
