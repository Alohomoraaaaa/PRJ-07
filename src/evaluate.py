"""
Model Evaluation and Error Analysis Module (Phase 6).
Computes Precision, Recall, F1, ROC-AUC, Confusion Matrix, and Feature Importances
across comparative models and feature regimes on entity-disjoint test data.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute standard binary classification evaluation metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else None

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "cm_raw": cm.tolist(),
    }


def evaluate_all_regimes(training_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate all trained models across experimental regimes on disjoint test sets.
    """
    eval_summary = {}

    for regime_name, res in training_results.items():
        y_test = res["y_test"]

        # 1. Logistic Regression Metrics
        lr_metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=res["lr"]["preds"],
            y_prob=res["lr"]["probs"],
        )

        # 2. Random Forest Metrics
        rf_metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=res["rf"]["preds"],
            y_prob=res["rf"]["probs"],
        )

        eval_summary[regime_name] = {
            "regime": regime_name,
            "feature_count": len(res["feature_cols"]),
            "features": res["feature_cols"],
            "logistic_regression": {
                "metrics": lr_metrics,
                "coefficients": res["lr"]["coefficients"],
            },
            "random_forest": {
                "metrics": rf_metrics,
                "importances": res["rf"]["importances"],
            },
        }

    return eval_summary


def format_evaluation_table(eval_summary: Dict[str, Any]) -> pd.DataFrame:
    """Create a formatted comparison DataFrame of model performance."""
    rows = []
    for regime_name, data in eval_summary.items():
        for model_name in ["logistic_regression", "random_forest"]:
            m = data[model_name]["metrics"]
            cm = m["confusion_matrix"]
            rows.append({
                "Feature Regime": regime_name,
                "Model": "Logistic Regression" if model_name == "logistic_regression" else "Random Forest",
                "Features (#)": data["feature_count"],
                "Precision": f"{m['precision']*100:.2f}%",
                "Recall": f"{m['recall']*100:.2f}%",
                "F1 Score": f"{m['f1_score']*100:.2f}%",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "TP": cm["true_positives"],
                "FP": cm["false_positives"],
                "FN": cm["false_negatives"],
                "TN": cm["true_negatives"],
            })

    return pd.DataFrame(rows)


def analyze_errors(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, pd.DataFrame]:
    """Identify False Positives and False Negatives with feature details."""
    df_with_preds = test_df.copy()
    df_with_preds["true_label"] = y_true
    df_with_preds["predicted_label"] = y_pred
    df_with_preds["match_probability"] = np.round(y_prob, 4)

    fps = df_with_preds[(df_with_preds["true_label"] == 0) & (df_with_preds["predicted_label"] == 1)]
    fns = df_with_preds[(df_with_preds["true_label"] == 1) & (df_with_preds["predicted_label"] == 0)]

    return {
        "false_positives": fps,
        "false_negatives": fns,
    }


if __name__ == "__main__":
    try:
        from src.train import run_all_training_experiments
    except ImportError:
        from train import run_all_training_experiments

    print("Executing Phase 5 & 6: Training and Disjoint Evaluation Pipeline...")
    training_results = run_all_training_experiments()
    eval_summary = evaluate_all_regimes(training_results)

    comp_table = format_evaluation_table(eval_summary)
    print("\n" + "=" * 90)
    print("MODEL PERFORMANCE COMPARISON ON ENTITY-DISJOINT TEST DATA")
    print("=" * 90)
    print(comp_table.to_string(index=False))

    print("\n" + "=" * 90)
    print("FEATURE IMPORTANCE / COEFFICIENT ANALYSIS")
    print("=" * 90)
    for regime_name, data in eval_summary.items():
        print(f"\n--- Regime: {regime_name.upper()} ---")
        rf_imp = pd.Series(data["random_forest"]["importances"]).sort_values(ascending=False).head(5)
        print("Top 5 Random Forest Features:")
        for feat, imp in rf_imp.items():
            print(f"  {feat:<25}: {imp:.4f} ({imp*100:.2f}%)")
