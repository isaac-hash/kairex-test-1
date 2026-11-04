"""
XGBoost Training Module for Kairex PSS SCALP ML Pipeline.
Performs purged walk-forward OOF estimation, Optuna hyperparameter tuning,
and trains the final model on the entire training set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss
import optuna

from src.config import SEED, MODELS_DIR, CAT_FEATURES
from src.data_loader import prepare_datasets
from src.validators import PurgedWalkForwardCV

optuna.logging.set_verbosity(optuna.logging.WARNING)

def format_xgb_data(X: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Ensures categorical columns are formatted as pd.Categorical for XGBoost."""
    X_xgb = X.copy()
    for col in cat_cols:
        if col in X_xgb.columns:
            X_xgb[col] = pd.Categorical(X_xgb[col].astype(str))
    return X_xgb

def run_xgboost_cv(X_train: pd.DataFrame, y_train: np.ndarray, train_df: pd.DataFrame, params: dict, cat_cols: list):
    """
    Runs purged walk-forward cross-validation for XGBoost with given parameters.
    Returns:
        oof_preds: np.ndarray of shape (len(train_df),)
        val_mask: boolean mask indicating which trades were evaluated out-of-fold
        fold_scores: list of dicts with fold-level metrics
    """
    cv = PurgedWalkForwardCV(n_folds=5, initial_train_pct=0.50)
    oof_preds = np.full(len(train_df), np.nan)
    fold_scores = []

    X_xgb = format_xgb_data(X_train, cat_cols)

    for tr_idx, val_idx, info in cv.split(train_df):
        X_tr, y_tr = X_xgb.iloc[tr_idx], y_train[tr_idx]
        X_val, y_val = X_xgb.iloc[val_idx], y_train[val_idx]

        model = XGBClassifier(
            **params,
            enable_categorical=True,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
            early_stopping_rounds=40,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds_val = model.predict_proba(X_val)[:, 1]
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

def tune_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, train_df: pd.DataFrame, cat_cols: list, n_trials: int = 25):
    """Tunes XGBoost hyperparameters using walk-forward CV AUC as objective."""
    print(f"\n[XGBoost] Tuning hyperparameters with Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 15.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        }
        oof, mask, _ = run_xgboost_cv(X_train, y_train, train_df, params, cat_cols)
        auc = roc_auc_score(y_train[mask], oof[mask])
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=240)

    print(f"[XGBoost] Best Trial AUC: {study.best_value:.4f}")
    print(f"[XGBoost] Best Params: {study.best_params}")
    return study.best_params

def train_xgboost_pipeline(tune: bool = True):
    """Full training pipeline for XGBoost."""
    print("=" * 60)
    print("XGBOOST TRAINING PIPELINE")
    print("=" * 60)

    train_df, test_df, X_train, y_train, X_test, y_test, feats = prepare_datasets()
    cat_cols = feats["cat_features"]

    default_params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 5.0,
        "reg_alpha": 0.1,
    }

    if tune:
        best_params = tune_xgboost(X_train, y_train, train_df, cat_cols, n_trials=25)
        params = {**default_params, **best_params}
    else:
        params = default_params

    # Run walk-forward CV with selected params
    print("\n[XGBoost] Running 5-fold Purged Walk-Forward Cross Validation...")
    oof_preds, val_mask, fold_scores = run_xgboost_cv(X_train, y_train, train_df, params, cat_cols)

    agg_auc = roc_auc_score(y_train[val_mask], oof_preds[val_mask])
    agg_loss = log_loss(y_train[val_mask], oof_preds[val_mask])
    print(f"[XGBoost OOF] Aggregated Walk-Forward AUC: {agg_auc:.4f} | LogLoss: {agg_loss:.4f}")
    for f in fold_scores:
        print(f"  Fold {f['fold']}: Val AUC = {f['auc']:.4f}, LogLoss = {f['logloss']:.4f} (Train: {f['n_train']}, Val: {f['n_val']})")

    # Fit final model on the entire training set
    print("\n[XGBoost] Fitting final model on full training set (951 samples)...")
    X_train_xgb = format_xgb_data(X_train, cat_cols)
    X_test_xgb = format_xgb_data(X_test, cat_cols)

    final_model = XGBClassifier(
        **params,
        enable_categorical=True,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
    )
    final_model.fit(X_train_xgb, y_train)

    # Predict on holdout test set
    test_preds = final_model.predict_proba(X_test_xgb)[:, 1]
    holdout_auc = roc_auc_score(y_test, test_preds)
    holdout_loss = log_loss(y_test, test_preds)
    print(f"[XGBoost Holdout 2026] AUC: {holdout_auc:.4f} | LogLoss: {holdout_loss:.4f}")

    # Save model and params
    model_path = MODELS_DIR / "xgboost_final.json"
    params_path = MODELS_DIR / "xgboost_params.json"
    final_model.save_model(str(model_path))
    with open(params_path, "w") as fp:
        json.dump(params, fp, indent=2)
    print(f"Saved model to: {model_path}")

    # Feature importance
    feature_imp = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

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
    train_xgboost_pipeline(tune=True)
