"""
Hybrid CatBoost-XGBoost Stacking Module for Kairex PSS SCALP ML Pipeline.
Implements a stacked model where Layer 0 consists of CatBoost and XGBoost,
and Layer 1 is a regularized meta-learner trained on Layer 0 Out-Of-Fold probabilities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, log_loss

from src.config import SEED, MODELS_DIR

def build_hybrid_model(oof_cat: np.ndarray, oof_xgb: np.ndarray, y_train: np.ndarray, val_mask: np.ndarray,
                       test_pred_cat: np.ndarray, test_pred_xgb: np.ndarray, y_test: np.ndarray):
    """
    Fits Layer 1 Meta-Learner on Out-Of-Fold probabilities of CatBoost and XGBoost.
    
    Args:
        oof_cat: OOF probability array for CatBoost (length = len(train_df))
        oof_xgb: OOF probability array for XGBoost (length = len(train_df))
        y_train: Ground truth binary labels for training set
        val_mask: Boolean mask indicating rows with valid OOF predictions
        test_pred_cat: Test set probabilities from full CatBoost model
        test_pred_xgb: Test set probabilities from full XGBoost model
        y_test: Ground truth binary labels for test set
    """
    print("=" * 60)
    print("HYBRID CATBOOST-XGBOOST STACKING PIPELINE")
    print("=" * 60)

    # Prepare Layer 0 OOF features for Meta-Learner
    X_meta_train = np.column_stack([oof_cat[val_mask], oof_xgb[val_mask]])
    y_meta_train = y_train[val_mask]

    print(f"\n[Hybrid] Fitting Meta-Learner on {len(X_meta_train)} OOF probability pairs...")
    meta_learner = LogisticRegressionCV(
        Cs=10,
        cv=5,
        scoring="roc_auc",
        random_state=SEED,
        max_iter=1000,
    )
    meta_learner.fit(X_meta_train, y_meta_train)

    coefs = meta_learner.coef_[0]
    intercept = meta_learner.intercept_[0]
    print(f"[Hybrid Meta-Learner] Intercept: {intercept:.4f}")
    print(f"[Hybrid Meta-Learner] Weights -> CatBoost: {coefs[0]:.4f}, XGBoost: {coefs[1]:.4f}")

    # Out-of-fold meta predictions
    oof_hybrid = np.full(len(oof_cat), np.nan)
    oof_hybrid[val_mask] = meta_learner.predict_proba(X_meta_train)[:, 1]

    agg_auc = roc_auc_score(y_meta_train, oof_hybrid[val_mask])
    agg_loss = log_loss(y_meta_train, oof_hybrid[val_mask])
    print(f"[Hybrid OOF] Aggregated Walk-Forward AUC: {agg_auc:.4f} | LogLoss: {agg_loss:.4f}")

    # Holdout predictions
    X_meta_test = np.column_stack([test_pred_cat, test_pred_xgb])
    test_preds = meta_learner.predict_proba(X_meta_test)[:, 1]

    holdout_auc = roc_auc_score(y_test, test_preds)
    holdout_loss = log_loss(y_test, test_preds)
    print(f"[Hybrid Holdout 2026] AUC: {holdout_auc:.4f} | LogLoss: {holdout_loss:.4f}")

    # Save meta-learner
    meta_path = MODELS_DIR / "hybrid_meta_learner.pkl"
    joblib.dump(meta_learner, str(meta_path))
    print(f"Saved hybrid meta-learner to: {meta_path}")

    return {
        "meta_learner": meta_learner,
        "oof_preds": oof_hybrid,
        "val_mask": val_mask,
        "test_preds": test_preds,
        "agg_auc": agg_auc,
        "agg_loss": agg_loss,
        "holdout_auc": holdout_auc,
        "holdout_loss": holdout_loss,
        "weights": {"catboost": float(coefs[0]), "xgboost": float(coefs[1]), "intercept": float(intercept)},
    }
