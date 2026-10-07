import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

def run():
    print("Loading dataset...")
    df = pd.read_csv("bedtime_screentime_sleep_debt.csv")
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # 1. Feature Groups
    habit_features = [
        "age", "gender", "occupation_type", "chronotype",
        "bedtime_phone_minutes", "primary_bedtime_app",
        "screen_brightness_pct", "blue_light_filter_active",
        "caffeine_post_5pm_mg", "physical_activity_min"
    ]
    categorical_features = ["gender", "occupation_type", "chronotype", "primary_bedtime_app"]
    numeric_features = [
        "age", "bedtime_phone_minutes", "screen_brightness_pct",
        "blue_light_filter_active", "caffeine_post_5pm_mg", "physical_activity_min"
    ]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
        ]
    )

    results = {
        "dataset_summary": {
            "total_records": int(len(df)),
            "features_count": int(df.shape[1]),
            "avg_bedtime_phone_min": round(float(df["bedtime_phone_minutes"].mean()), 1),
            "avg_sleep_latency_min": round(float(df["sleep_latency_min"].mean()), 1),
            "avg_sleep_hours": round(float(df["total_sleep_hours"].mean()), 2),
            "avg_fatigue_score": round(float(df["next_day_fatigue_score"].mean()), 2),
            "debt_categories": df["sleep_debt_category"].value_counts().to_dict(),
            "top_bedtime_apps": df["primary_bedtime_app"].value_counts().to_dict(),
            "chronotypes": df["chronotype"].value_counts().to_dict()
        }
    }

    # =========================================================================
    # PROJECT IDEA 3: Sleep Debt Category Classifier (Multi-class Classification)
    # =========================================================================
    print("\n--- Training Model 1: Sleep Debt Category Classifier ---")
    # Using habit features + sleep hours + alarm snoozes to predict category
    cls_features = habit_features + ["total_sleep_hours", "morning_alarm_snoozes"]
    cls_num_features = numeric_features + ["total_sleep_hours", "morning_alarm_snoozes"]
    
    cls_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cls_num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
        ]
    )

    X_cls = df[cls_features]
    y_cls = df["sleep_debt_category"]
    
    classes_list = sorted(list(y_cls.unique()))
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    # Train Random Forest Classifier
    rf_cls = Pipeline([
        ("prep", cls_preprocessor),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12))
    ])
    rf_cls.fit(X_train_c, y_train_c)
    y_pred_c = rf_cls.predict(X_test_c)

    acc = accuracy_score(y_test_c, y_pred_c)
    cm = confusion_matrix(y_test_c, y_pred_c, labels=classes_list).tolist()
    cr = classification_report(y_test_c, y_pred_c, output_dict=True)

    print(f"Classification Accuracy: {acc:.4f}")

    # Extract feature names & importances
    encoder = rf_cls.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(encoder.get_feature_names_out(categorical_features))
    all_cls_feature_names = cls_num_features + cat_feature_names
    importances_cls = rf_cls.named_steps["rf"].feature_importances_
    
    cls_feat_imp = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(all_cls_feature_names, importances_cls)],
        key=lambda x: x["importance"], reverse=True
    )[:8]

    results["classification_sleep_debt"] = {
        "model_name": "Random Forest Classifier",
        "accuracy": round(float(acc) * 100, 2),
        "classes": classes_list,
        "confusion_matrix": cm,
        "metrics": {
            "macro_f1": round(float(cr["macro avg"]["f1-score"]), 3),
            "weighted_f1": round(float(cr["weighted avg"]["f1-score"]), 3)
        },
        "top_features": cls_feat_imp
    }

    # =========================================================================
    # PROJECT IDEA 4 (A): Sleep Latency Predictor (Regression)
    # Target: sleep_latency_min (minutes to fall asleep) based on bedtime habits
    # =========================================================================
    print("\n--- Training Model 2: Sleep Latency Predictor (Minutes) ---")
    X_lat = df[habit_features]
    y_lat = df["sleep_latency_min"]

    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_lat, y_lat, test_size=0.2, random_state=42
    )

    reg_latency = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
    ])
    reg_latency.fit(X_train_l, y_train_l)
    y_pred_lat = reg_latency.predict(X_test_l)

    mae_lat = mean_absolute_error(y_test_l, y_pred_lat)
    rmse_lat = np.sqrt(mean_squared_error(y_test_l, y_pred_lat))
    r2_lat = r2_score(y_test_l, y_pred_lat)

    print(f"Sleep Latency R2: {r2_lat:.4f}, MAE: {mae_lat:.2f} min, RMSE: {rmse_lat:.2f} min")

    encoder_lat = reg_latency.named_steps["prep"].named_transformers_["cat"]
    all_lat_feature_names = numeric_features + list(encoder_lat.get_feature_names_out(categorical_features))
    importances_lat = reg_latency.named_steps["rf"].feature_importances_
    
    lat_feat_imp = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(all_lat_feature_names, importances_lat)],
        key=lambda x: x["importance"], reverse=True
    )[:8]

    # Also fit a linear baseline to easily provide lightweight browser inference coefficients
    ridge_lat = Pipeline([
        ("prep", preprocessor),
        ("reg", Ridge(alpha=1.0))
    ])
    ridge_lat.fit(X_lat, y_lat)

    results["regression_sleep_latency"] = {
        "model_name": "Random Forest Regressor",
        "r2_score": round(float(r2_lat), 4),
        "mae_minutes": round(float(mae_lat), 2),
        "rmse_minutes": round(float(rmse_lat), 2),
        "top_features": lat_feat_imp
    }

    # =========================================================================
    # PROJECT IDEA 4 (B): Next-Day Fatigue Score Predictor (Regression)
    # Target: next_day_fatigue_score (1.0 to 10.0 scale)
    # =========================================================================
    print("\n--- Training Model 3: Next-Day Fatigue Predictor (Score 1-10) ---")
    fatigue_features = habit_features + ["total_sleep_hours", "morning_alarm_snoozes"]
    X_fat = df[fatigue_features]
    y_fat = df["next_day_fatigue_score"]

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_fat, y_fat, test_size=0.2, random_state=42
    )

    reg_fatigue = Pipeline([
        ("prep", cls_preprocessor),
        ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5))
    ])
    reg_fatigue.fit(X_train_f, y_train_f)
    y_pred_fat = reg_fatigue.predict(X_test_f)

    mae_fat = mean_absolute_error(y_test_f, y_pred_fat)
    rmse_fat = np.sqrt(mean_squared_error(y_test_f, y_pred_fat))
    r2_fat = r2_score(y_test_f, y_pred_fat)

    print(f"Fatigue Score R2: {r2_fat:.4f}, MAE: {mae_fat:.2f}, RMSE: {rmse_fat:.2f}")

    encoder_fat = reg_fatigue.named_steps["prep"].named_transformers_["cat"]
    all_fat_feature_names = cls_num_features + list(encoder_fat.get_feature_names_out(categorical_features))
    importances_fat = reg_fatigue.named_steps["gb"].feature_importances_
    
    fat_feat_imp = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(all_fat_feature_names, importances_fat)],
        key=lambda x: x["importance"], reverse=True
    )[:8]

    results["regression_fatigue_score"] = {
        "model_name": "Gradient Boosting Regressor",
        "r2_score": round(float(r2_fat), 4),
        "mae_score": round(float(mae_fat), 2),
        "rmse_score": round(float(rmse_fat), 2),
        "top_features": fat_feat_imp
    }

    # Save summary to JSON
    with open("model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved evaluation results to model_results.json")

if __name__ == "__main__":
    run()
