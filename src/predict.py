"""
ML Prediction and Inference Module (Phase 7).
Loads trained model artifacts (e.g. Realistic No-SSN Random Forest),
runs inference on candidate record pairs, and computes match probabilities and labels.
"""

import os
import joblib
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


class EntityMatchPredictor:
    """
    Inference engine for predicting record pair linkage.
    """

    def __init__(
        self,
        model_path: str = "models/rf_no_ssn.joblib",
        feature_regime: str = "no_ssn",
        classification_threshold: float = 0.60,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Train models first.")

        self.model = joblib.load(model_path)
        self.feature_regime = feature_regime
        self.threshold = classification_threshold

        # Determine feature columns based on regime
        from src.train import SSN_FEATURES, DOB_FEATURES, EXCLUDED_COLUMNS

        self.ssn_features = SSN_FEATURES
        self.dob_features = DOB_FEATURES
        self.excluded_cols = EXCLUDED_COLUMNS

    def filter_features(self, feature_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Filter DataFrame columns to match the model's feature set."""
        all_cols = list(feature_df.columns)
        candidate_cols = [c for c in all_cols if c not in self.excluded_cols]

        if self.feature_regime == "no_ssn":
            active_cols = [c for c in candidate_cols if c not in self.ssn_features]
        elif self.feature_regime == "demographic_only":
            active_cols = [c for c in candidate_cols if c not in self.ssn_features and c not in self.dob_features]
        else:
            active_cols = candidate_cols

        X = feature_df[active_cols].values
        return X, active_cols

    def predict_candidate_pairs(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on feature DataFrame of candidate pairs.
        Returns DataFrame with pair IDs, match probability, prediction, and confidence score.
        """
        X, active_cols = self.filter_features(feature_df)

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        results_df = pd.DataFrame()
        results_df["rec_id_a"] = feature_df["rec_id_a"]
        results_df["rec_id_b"] = feature_df["rec_id_b"]
        results_df["match_probability"] = np.round(probs, 4)
        results_df["predicted_match"] = preds
        results_df["confidence_score"] = np.round(np.abs(probs - 0.5) * 2.0, 4)  # 0.0 to 1.0 confidence

        if "match_label" in feature_df.columns:
            results_df["ground_truth_label"] = feature_df["match_label"]

        if "num_passes_triggered" in feature_df.columns:
            results_df["num_passes_triggered"] = feature_df["num_passes_triggered"]

        return results_df


def run_prediction_pipeline(
    processed_dir: str = "data/processed",
    model_path: str = "models/rf_no_ssn.joblib",
    classification_threshold: float = 0.60,
    output_path: Optional[str] = "data/processed/match_predictions.csv",
) -> pd.DataFrame:
    """Execute end-to-end prediction on full candidate features."""
    features_path = os.path.join(processed_dir, "candidate_features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Candidate features not found at: {features_path}")

    features_df = pd.read_csv(features_path)
    predictor = EntityMatchPredictor(
        model_path=model_path,
        feature_regime="no_ssn",
        classification_threshold=classification_threshold,
    )

    predictions_df = predictor.predict_candidate_pairs(features_df)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)

    return predictions_df


if __name__ == "__main__":
    preds = run_prediction_pipeline()
    print("Inference Complete:")
    print(f"  Total Candidates Evaluated: {len(preds):,}")
    print(f"  Predicted Matches:          {(preds['predicted_match'] == 1).sum():,}")
    print(f"  Predicted Non-Matches:      {(preds['predicted_match'] == 0).sum():,}")
    print(f"  Mean Match Probability:     {preds['match_probability'].mean():.4f}")
