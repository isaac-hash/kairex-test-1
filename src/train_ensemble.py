"""
Ensemble Techniques Module for Kairex PSS SCALP ML Pipeline.
Implements:
1. Simple Probability Averaging
2. Optuna-Weighted Averaging
3. Rank-Based Percentile Ensemble
4. Hard Voting Ensemble
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, log_loss
import optuna

from src.config import SEED

def rank_transform(probs: np.ndarray) -> np.ndarray:
    """Converts a 1D probability array into empirical percentile ranks [0, 1]."""
    return (rankdata(probs) - 1.0) / (len(probs) - 1.0 + 1e-9)

def build_ensembles(oof_cat: np.ndarray, oof_xgb: np.ndarray, y_train: np.ndarray, val_mask: np.ndarray,
                    test_cat: np.ndarray, test_xgb: np.ndarray, y_test: np.ndarray):
    """
    Constructs multiple ensemble techniques and evaluates them on OOF and Holdout.
    """
    print("=" * 60)
    print("ENSEMBLE TECHNIQUES PIPELINE")
    print("=" * 60)

    y_val = y_train[val_mask]
    p_cat_val = oof_cat[val_mask]
    p_xgb_val = oof_xgb[val_mask]

    # 1. Simple Average
    oof_simple = np.full(len(oof_cat), np.nan)
    oof_simple[val_mask] = (p_cat_val + p_xgb_val) / 2.0
    test_simple = (test_cat + test_xgb) / 2.0

    # 2. Weighted Average via Optuna optimization on OOF
    print("\n[Ensemble] Optimizing combination weights on OOF...")
    def objective(trial):
        w = trial.suggest_float("w_cat", 0.0, 1.0)
        p = w * p_cat_val + (1.0 - w) * p_xgb_val
        return roc_auc_score(y_val, p)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=50)

    best_w = study.best_params["w_cat"]
    print(f"[Ensemble] Optimal weights: CatBoost = {best_w:.3f}, XGBoost = {1.0 - best_w:.3f}")

    oof_weighted = np.full(len(oof_cat), np.nan)
    oof_weighted[val_mask] = best_w * p_cat_val + (1.0 - best_w) * p_xgb_val
    test_weighted = best_w * test_cat + (1.0 - best_w) * test_xgb

    # 3. Rank-Based Ensemble
    oof_rank = np.full(len(oof_cat), np.nan)
    oof_rank[val_mask] = (rank_transform(p_cat_val) + rank_transform(p_xgb_val)) / 2.0
    test_rank = (rank_transform(test_cat) + rank_transform(test_xgb)) / 2.0

    # 4. Hard Voting (Fraction of models predicting >= 0.50)
    oof_vote = np.full(len(oof_cat), np.nan)
    oof_vote[val_mask] = ((p_cat_val >= 0.50).astype(float) + (p_xgb_val >= 0.50).astype(float)) / 2.0
    test_vote = ((test_cat >= 0.50).astype(float) + (test_xgb >= 0.50).astype(float)) / 2.0

    ensembles = {
        "Ensemble_Simple": {
            "name": "Simple Average",
            "oof": oof_simple,
            "test": test_simple,
            "oof_auc": roc_auc_score(y_val, oof_simple[val_mask]),
            "oof_loss": log_loss(y_val, oof_simple[val_mask]),
            "test_auc": roc_auc_score(y_test, test_simple),
            "test_loss": log_loss(y_test, test_simple),
        },
        "Ensemble_Weighted": {
            "name": f"Weighted Average (w_cat={best_w:.2f})",
            "oof": oof_weighted,
            "test": test_weighted,
            "oof_auc": roc_auc_score(y_val, oof_weighted[val_mask]),
            "oof_loss": log_loss(y_val, oof_weighted[val_mask]),
            "test_auc": roc_auc_score(y_test, test_weighted),
            "test_loss": log_loss(y_test, test_weighted),
            "weights": {"w_cat": best_w, "w_xgb": 1.0 - best_w},
        },
        "Ensemble_Rank": {
            "name": "Rank-Based Ensemble",
            "oof": oof_rank,
            "test": test_rank,
            "oof_auc": roc_auc_score(y_val, oof_rank[val_mask]),
            "oof_loss": log_loss(y_val, oof_rank[val_mask]),
            "test_auc": roc_auc_score(y_test, test_rank),
            "test_loss": log_loss(y_test, test_rank),
        },
        "Ensemble_Voting": {
            "name": "Hard Voting",
            "oof": oof_vote,
            "test": test_vote,
            "oof_auc": roc_auc_score(y_val, oof_vote[val_mask]),
            "oof_loss": log_loss(y_val, np.clip(oof_vote[val_mask], 1e-6, 1 - 1e-6)),
            "test_auc": roc_auc_score(y_test, test_vote),
            "test_loss": log_loss(y_test, np.clip(test_vote, 1e-6, 1 - 1e-6)),
        },
    }

    print("\n[Ensemble Results Summary]")
    for k, v in ensembles.items():
        print(f"  {v['name']:<30s} | OOF AUC: {v['oof_auc']:.4f} (Loss: {v['oof_loss']:.4f}) | Test AUC: {v['test_auc']:.4f} (Loss: {v['test_loss']:.4f})")

    return ensembles
