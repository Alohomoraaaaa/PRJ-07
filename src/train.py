"""
Leakage-Safe Model Training Module (Phase 5).
Trains interpretable supervised baseline models (Logistic Regression & Random Forest)
on entity-disjoint training features and evaluates under different information-availability regimes:
  1. Full Identity (All features including SSN and DOB)
  2. Realistic / No-SSN (Excluding ssn_* features)
  3. Pure Demographic (Excluding ssn_* and dob_* features)
"""

import os
import joblib
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


# Feature definitions
SSN_FEATURES = ["ssn_exact", "ssn_lev"]
DOB_FEATURES = ["dob_exact", "dob_lev", "dob_year_match"]

EXCLUDED_COLUMNS = ["rec_id_a", "rec_id_b", "match_label"]


def get_feature_subsets(all_columns: List[str]) -> Dict[str, List[str]]:
    """Return configured feature subsets for comparative experiments."""
    candidate_features = [c for c in all_columns if c not in EXCLUDED_COLUMNS]

    full_features = candidate_features
    no_ssn_features = [c for c in candidate_features if c not in SSN_FEATURES]
    demographic_features = [c for c in candidate_features if c not in SSN_FEATURES and c not in DOB_FEATURES]

    return {
        "full_identity": full_features,
        "no_ssn": no_ssn_features,
        "demographic_only": demographic_features,
    }


def train_models_on_partition(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    regime_name: str = "full_identity",
    models_dir: str = "models",
) -> Dict[str, Any]:
    """
    Train Logistic Regression and Random Forest models on the specified feature set.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    os.makedirs(models_dir, exist_ok=True)

    X_train = train_df[feature_cols].values
    y_train = train_df["match_label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["match_label"].values

    # Train Logistic Regression (with standard scaler for coefficients interpretability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_model.fit(X_train_scaled, y_train)

    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Save LR model & scaler
    joblib.dump(lr_model, os.path.join(models_dir, f"lr_{regime_name}.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, f"scaler_{regime_name}.joblib"))

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    # Save RF model
    joblib.dump(rf_model, os.path.join(models_dir, f"rf_{regime_name}.joblib"))

    return {
        "regime_name": regime_name,
        "feature_cols": feature_cols,
        "y_test": y_test,
        "lr": {
            "model": lr_model,
            "scaler": scaler,
            "preds": lr_preds,
            "probs": lr_probs,
            "coefficients": dict(zip(feature_cols, lr_model.coef_[0])),
            "intercept": float(lr_model.intercept_[0]),
        },
        "rf": {
            "model": rf_model,
            "preds": rf_preds,
            "probs": rf_probs,
            "importances": dict(zip(feature_cols, rf_model.feature_importances_)),
        },
    }


def run_all_training_experiments(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
) -> Dict[str, Any]:
    """
    Run comparative model training across all experimental feature regimes:
      1. Full Identity (with SSN & DOB)
      2. No-SSN (Simulating missing/unshared National ID)
      3. Demographic Only (Simulating noisy demographic linkage)
    """
    train_path = os.path.join(processed_dir, "train_features.csv")
    test_path = os.path.join(processed_dir, "test_features.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Feature datasets not found. Run Phase 4 first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_subsets = get_feature_subsets(list(train_df.columns))

    results = {}
    for regime_name, feats in feature_subsets.items():
        print(f"Training models for regime: '{regime_name}' ({len(feats)} features)...")
        res = train_models_on_partition(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feats,
            regime_name=regime_name,
            models_dir=models_dir,
        )
        results[regime_name] = res

    return results


if __name__ == "__main__":
    results = run_all_training_experiments()
    print("All models trained and saved to models/ directory.")
