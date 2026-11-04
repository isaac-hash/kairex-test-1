"""
CatBoost Training Module for Kairex PSS SCALP ML Pipeline.
Performs purged walk-forward OOF estimation, Optuna hyperparameter tuning,
and trains the final model on the entire training set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, log_loss
import optuna

from src.config import SEED, MODELS_DIR, CAT_FEATURES
from src.data_loader import prepare_datasets
from src.validators import PurgedWalkForwardCV

optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_cat_indices(X: pd.DataFrame, cat_cols: list) -> list:
    """Returns integer column indices for categorical features."""
    return [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

def run_catboost_cv(X_train: pd.DataFrame, y_train: np.ndarray, train_df: pd.DataFrame, params: dict, cat_cols: list):
    """
    Runs purged walk-forward cross-validation for CatBoost with given parameters.
    Returns:
        oof_preds: np.ndarray of shape (len(train_df),) with NaN for unvalidated first half
        val_mask: boolean mask indicating which trades were evaluated out-of-fold
        fold_scores: list of dicts with fold-level metrics
    """
    cv = PurgedWalkForwardCV(n_folds=5, initial_train_pct=0.50)
    oof_preds = np.full(len(train_df), np.nan)
    cat_idx = get_cat_indices(X_train, cat_cols)
    fold_scores = []

    for tr_idx, val_idx, info in cv.split(train_df):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        val_pool = Pool(X_val, y_val, cat_features=cat_idx)

        model = CatBoostClassifier(**params, cat_features=cat_idx, random_seed=SEED, verbose=0, thread_count=-1)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=40, verbose=0)

        preds_val = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = preds_val

        auc = roc_auc_score(y_val, preds_val) if len(np.unique(y_val)) > 1 else 0.5
        loss = log_loss(y_val, preds_val)

        fold_scores.append({
            "fold": info["fold"],
            "auc": float(auc),
            "logloss": float(loss),
            "n_train": info["n_train"],
            "n_val": info["n_val"],
        })

    val_mask = ~np.isnan(oof_preds)
    return oof_preds, val_mask, fold_scores

def tune_catboost(X_train: pd.DataFrame, y_train: np.ndarray, train_df: pd.DataFrame, cat_cols: list, n_trials: int = 25):
    """Tunes CatBoost hyperparameters using walk-forward CV AUC as objective."""
    print(f"\n[CatBoost] Tuning hyperparameters with Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600, step=50),
            "depth": trial.suggest_int("depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 15.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "eval_metric": "Logloss",
        }
        oof, mask, _ = run_catboost_cv(X_train, y_train, train_df, params, cat_cols)
        auc = roc_auc_score(y_train[mask], oof[mask])
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=240)

    print(f"[CatBoost] Best Trial AUC: {study.best_value:.4f}")
    print(f"[CatBoost] Best Params: {study.best_params}")
    return study.best_params

def train_catboost_pipeline(tune: bool = True):
    """Full training pipeline for CatBoost."""
    print("=" * 60)
    print("CATBOOST TRAINING PIPELINE")
    print("=" * 60)

    train_df, test_df, X_train, y_train, X_test, y_test, feats = prepare_datasets()
    cat_cols = feats["cat_features"]

    default_params = {
        "iterations": 400,
        "depth": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "eval_metric": "Logloss",
    }

    if tune:
        best_params = tune_catboost(X_train, y_train, train_df, cat_cols, n_trials=25)
        params = {**default_params, **best_params}
    else:
        params = default_params

    # Run walk-forward CV with selected params
    print("\n[CatBoost] Running 5-fold Purged Walk-Forward Cross Validation...")
    oof_preds, val_mask, fold_scores = run_catboost_cv(X_train, y_train, train_df, params, cat_cols)

    agg_auc = roc_auc_score(y_train[val_mask], oof_preds[val_mask])
    agg_loss = log_loss(y_train[val_mask], oof_preds[val_mask])
    print(f"[CatBoost OOF] Aggregated Walk-Forward AUC: {agg_auc:.4f} | LogLoss: {agg_loss:.4f}")
    for f in fold_scores:
        print(f"  Fold {f['fold']}: Val AUC = {f['auc']:.4f}, LogLoss = {f['logloss']:.4f} (Train: {f['n_train']}, Val: {f['n_val']})")

    # Fit final model on the entire training set
    print("\n[CatBoost] Fitting final model on full training set (951 samples)...")
    cat_idx = get_cat_indices(X_train, cat_cols)
    full_train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool = Pool(X_test, y_test, cat_features=cat_idx)

    final_model = CatBoostClassifier(**params, cat_features=cat_idx, random_seed=SEED, verbose=0, thread_count=-1)
    final_model.fit(full_train_pool)

    # Predict on holdout test set
    test_preds = final_model.predict_proba(test_pool)[:, 1]
    holdout_auc = roc_auc_score(y_test, test_preds)
    holdout_loss = log_loss(y_test, test_preds)
    print(f"[CatBoost Holdout 2026] AUC: {holdout_auc:.4f} | LogLoss: {holdout_loss:.4f}")

    # Save model and params
    model_path = MODELS_DIR / "catboost_final.cbm"
    params_path = MODELS_DIR / "catboost_params.json"
    final_model.save_model(str(model_path))
    with open(params_path, "w") as fp:
        json.dump(params, fp, indent=2)
    print(f"Saved model to: {model_path}")

    # Feature importance
    feature_imp = pd.Series(final_model.get_feature_importance(), index=X_train.columns).sort_values(ascending=False)

    return {
        "model": final_model,
        "params": params,
        "oof_preds": oof_preds,
        "val_mask": val_mask,
        "fold_scores": fold_scores,
        "test_preds": test_preds,
        "agg_auc": agg_auc,
        "agg_loss": agg_loss,
        "holdout_auc": holdout_auc,
        "holdout_loss": holdout_loss,
        "feature_importance": feature_imp,
    }

if __name__ == "__main__":
    train_catboost_pipeline(tune=True)
