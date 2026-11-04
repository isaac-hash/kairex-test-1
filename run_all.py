"""
Master Orchestration Script for Kairex PSS SCALP ML Pipeline.
Executes the full pipeline end-to-end:
1. Data ingestion, validation, and feature engineering
2. CatBoost training + tuning + walk-forward OOF estimation
3. XGBoost training + tuning + walk-forward OOF estimation
4. Hybrid CatBoost-XGBoost stacking meta-learner
5. Ensemble techniques (Simple, Weighted, Rank-Based, Voting)
6. Comprehensive evaluation, simulation, charts, and report generation
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.data_loader import prepare_datasets
from src.train_catboost import train_catboost_pipeline
from src.train_xgboost import train_xgboost_pipeline
from src.train_hybrid import build_hybrid_model
from src.train_ensemble import build_ensembles
from src.evaluate import evaluate_all
from src.config import REPORTS_DIR, PLOTS_DIR, PREDICTIONS_DIR, MODELS_DIR

def main():
    t0 = time.time()
    print("=" * 80)
    print("KAIREX PSS SCALP ML PIPELINE — EXECUTION STARTED")
    print("=" * 80)

    # 1. Prepare data
    train_df, test_df, X_train, y_train, X_test, y_test, feats = prepare_datasets(dedup_test=True)

    # 2. CatBoost
    cb_results = train_catboost_pipeline(tune=True)

    # 3. XGBoost
    xgb_results = train_xgboost_pipeline(tune=True)

    # 4. Hybrid Stacking
    hybrid_results = build_hybrid_model(
        oof_cat=cb_results["oof_preds"],
        oof_xgb=xgb_results["oof_preds"],
        y_train=y_train,
        val_mask=cb_results["val_mask"],
        test_pred_cat=cb_results["test_preds"],
        test_pred_xgb=xgb_results["test_preds"],
        y_test=y_test,
    )

    # 5. Ensemble Techniques
    ensemble_results = build_ensembles(
        oof_cat=cb_results["oof_preds"],
        oof_xgb=xgb_results["oof_preds"],
        y_train=y_train,
        val_mask=cb_results["val_mask"],
        test_cat=cb_results["test_preds"],
        test_xgb=xgb_results["test_preds"],
        y_test=y_test,
    )

    # 6. Evaluation, Simulation, Charts, and Markdown Reports
    eval_results = evaluate_all(
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        y_test=y_test,
        val_mask=cb_results["val_mask"],
        cb_res=cb_results,
        xgb_res=xgb_results,
        hybrid_res=hybrid_results,
        ens_res=ensemble_results,
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS")
    print("=" * 80)

    print("\n[Generated Artifacts Overview]")
    print(f"  Models:      {MODELS_DIR}")
    print(f"  Predictions: {PREDICTIONS_DIR}")
    print(f"  Charts:      {PLOTS_DIR}")
    print(f"  Reports:     {REPORTS_DIR}")

    print("\nReports created in outputs/reports/:")
    for r in sorted(REPORTS_DIR.glob("*.md")):
        print(f"  - {r.name}")

if __name__ == "__main__":
    main()
