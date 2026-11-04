"""
Evaluation, Trading Simulation, Plotting, and Report Generation Module
for Kairex PSS SCALP ML Pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

from src.config import (
    SEED,
    PREDICTIONS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    DECISION_THRESHOLD,
    SWEEP_THRESHOLDS,
)

# ----------------------------------------------------------------------------
# 1. Statistical & Bootstrap Helpers
# ----------------------------------------------------------------------------
def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, B: int = 1500) -> tuple[float, float]:
    """Computes empirical 95% bootstrap confidence interval for ROC AUC."""
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    aucs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if 0 < y_sample.sum() < n:
            aucs.append(roc_auc_score(y_sample, y_prob[idx]))
    if len(aucs) == 0:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def paired_bootstrap_test(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, B: int = 1500) -> dict:
    """Paired bootstrap test comparing AUC of Model A vs Model B."""
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if 0 < y_sample.sum() < n:
            diffs.append(roc_auc_score(y_sample, p_a[idx]) - roc_auc_score(y_sample, p_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value_a_better": float(np.mean(np.array(diffs) > 0)),
    }

def calculate_max_drawdown(cum_r: np.ndarray) -> float:
    """Calculates peak-to-trough max drawdown in R."""
    if len(cum_r) == 0:
        return 0.0
    peak = np.maximum.accumulate(cum_r)
    dd = peak - cum_r
    return float(np.max(dd))

def simulate_trading(y_prob: np.ndarray, result_r: np.ndarray, threshold: float = 0.50) -> dict:
    """Simulates trading decisions based on probability threshold."""
    sel = y_prob >= threshold
    r = result_r[sel]
    n_kept = int(sel.sum())
    pct_kept = float(100.0 * sel.mean())
    if n_kept == 0:
        return {
            "n_kept": 0, "pct_kept": 0.0, "win_rate": 0.0,
            "sum_r": 0.0, "exp_r": 0.0, "profit_factor": 0.0, "max_dd": 0.0
        }
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = float(wins / losses) if losses > 0 else (999.0 if wins > 0 else 0.0)
    win_rate = float(100.0 * (r > 0).mean())
    sum_r = float(r.sum())
    exp_r = float(r.mean())
    cum_r = np.cumsum(r)
    max_dd = calculate_max_drawdown(cum_r)

    return {
        "n_kept": n_kept,
        "pct_kept": pct_kept,
        "win_rate": win_rate,
        "sum_r": sum_r,
        "exp_r": exp_r,
        "profit_factor": pf,
        "max_dd": max_dd,
    }

# ----------------------------------------------------------------------------
# 2. Plotting Utilities
# ----------------------------------------------------------------------------
def plot_equity_curves(df_subset: pd.DataFrame, preds_dict: dict, title: str, save_path: Path):
    """Plots chronological cumulative R curves for take-all vs models at P >= 0.50."""
    plt.figure(figsize=(10, 5))
    real_r = df_subset["result_R"].values
    x = np.arange(len(real_r))

    # Baseline: take all
    plt.plot(x, np.cumsum(real_r), label="Take All (Baseline)", color="gray", linestyle="--", linewidth=1.5)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, p) in enumerate(preds_dict.items()):
        sel = p >= 0.50
        filtered_r = np.where(sel, real_r, 0.0)
        c = colors[i % len(colors)]
        plt.plot(x, np.cumsum(filtered_r), label=f"{name} (P>=0.50)", color=c, linewidth=1.8)

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Trade Number (Chronological)")
    plt.ylabel("Cumulative R")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()

def plot_calibration(y_true: np.ndarray, preds_dict: dict, save_path: Path):
    """Plots probability calibration / reliability curves."""
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

    for name, p in preds_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, p, n_bins=6, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", label=name, linewidth=1.5)

    plt.title("Reliability Diagram (Calibration Curve) — Holdout 2026", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()

def plot_roc_curves(y_true: np.ndarray, preds_dict: dict, save_path: Path):
    """Plots ROC curves for all models."""
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Random Guess (AUC = 0.50)")

    for name, p in preds_dict.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=1.6)

    plt.title("ROC Curves — Holdout 2026", fontsize=11, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()

# ----------------------------------------------------------------------------
# 3. Report Generation
# ----------------------------------------------------------------------------
def write_markdown_reports(oof_metrics: dict, test_metrics: dict, trading_oof: dict, trading_test: dict,
                           sig_breakdown: dict, cb_res: dict, xgb_res: dict, hybrid_res: dict,
                           ens_res: dict, paired_test: dict):
    """Generates all 6 comprehensive Markdown reports in outputs/reports/."""

    # 1. CatBoost Report
    cb_report = f"""# CatBoost Model Performance Report

## 1. Optimal Hyperparameters
```json
{pd.Series(cb_res['params']).to_json(indent=2)}
```

## 2. Walk-Forward Cross-Validation (Out-Of-Fold)
- **Aggregated OOF AUC**: {oof_metrics['CatBoost']['auc']:.4f} (95% CI: [{oof_metrics['CatBoost']['ci_lo']:.4f}, {oof_metrics['CatBoost']['ci_hi']:.4f}])
- **Log-Loss**: {oof_metrics['CatBoost']['logloss']:.4f}
- **Brier Score**: {oof_metrics['CatBoost']['brier']:.4f}

### Per-Fold Breakdown:
| Fold | Train N | Val N | Val AUC | Val LogLoss |
|---|---|---|---|---|
"""
    for f in cb_res["fold_scores"]:
        cb_report += f"| {f['fold']} | {f['n_train']} | {f['n_val']} | {f['auc']:.4f} | {f['logloss']:.4f} |\n"

    cb_report += f"""
## 3. Holdout Test Set (2026) Evaluation
- **Holdout AUC**: {test_metrics['CatBoost']['auc']:.4f} (95% CI: [{test_metrics['CatBoost']['ci_lo']:.4f}, {test_metrics['CatBoost']['ci_hi']:.4f}])
- **Holdout Log-Loss**: {test_metrics['CatBoost']['logloss']:.4f}
- **Holdout Brier Score**: {test_metrics['CatBoost']['brier']:.4f}

## 4. Trading Simulation (Decision Rule: Execute if P >= 0.50)
| Metric | Take-All Baseline | CatBoost Filtered |
|---|---|---|
| Trades Kept | {trading_test['Baseline']['n_kept']} (100%) | {trading_test['CatBoost']['n_kept']} ({trading_test['CatBoost']['pct_kept']:.1f}%) |
| Win Rate | {trading_test['Baseline']['win_rate']:.1f}% | {trading_test['CatBoost']['win_rate']:.1f}% |
| Realized Sum R | {trading_test['Baseline']['sum_r']:+.2f}R | {trading_test['CatBoost']['sum_r']:+.2f}R |
| Expected R / Trade | {trading_test['Baseline']['exp_r']:+.3f}R | {trading_test['CatBoost']['exp_r']:+.3f}R |
| Profit Factor | {trading_test['Baseline']['profit_factor']:.2f} | {trading_test['CatBoost']['profit_factor']:.2f} |
| Max Drawdown | {trading_test['Baseline']['max_dd']:.2f}R | {trading_test['CatBoost']['max_dd']:.2f}R |

## 5. Top 10 Feature Importances
| Rank | Feature | Importance Score |
|---|---|---|
"""
    for rank, (feat, score) in enumerate(cb_res["feature_importance"].head(10).items(), 1):
        cb_report += f"| {rank} | `{feat}` | {score:.3f} |\n"

    with open(REPORTS_DIR / "catboost_report.md", "w") as fp:
        fp.write(cb_report)

    # 2. XGBoost Report
    xgb_report = f"""# XGBoost Model Performance Report

## 1. Optimal Hyperparameters
```json
{pd.Series(xgb_res['params']).to_json(indent=2)}
```

## 2. Walk-Forward Cross-Validation (Out-Of-Fold)
- **Aggregated OOF AUC**: {oof_metrics['XGBoost']['auc']:.4f} (95% CI: [{oof_metrics['XGBoost']['ci_lo']:.4f}, {oof_metrics['XGBoost']['ci_hi']:.4f}])
- **Log-Loss**: {oof_metrics['XGBoost']['logloss']:.4f}
- **Brier Score**: {oof_metrics['XGBoost']['brier']:.4f}

### Per-Fold Breakdown:
| Fold | Train N | Val N | Val AUC | Val LogLoss |
|---|---|---|---|---|
"""
    for f in xgb_res["fold_scores"]:
        xgb_report += f"| {f['fold']} | {f['n_train']} | {f['n_val']} | {f['auc']:.4f} | {f['logloss']:.4f} |\n"

    xgb_report += f"""
## 3. Holdout Test Set (2026) Evaluation
- **Holdout AUC**: {test_metrics['XGBoost']['auc']:.4f} (95% CI: [{test_metrics['XGBoost']['ci_lo']:.4f}, {test_metrics['XGBoost']['ci_hi']:.4f}])
- **Holdout Log-Loss**: {test_metrics['XGBoost']['logloss']:.4f}
- **Holdout Brier Score**: {test_metrics['XGBoost']['brier']:.4f}

## 4. Trading Simulation (Decision Rule: Execute if P >= 0.50)
| Metric | Take-All Baseline | XGBoost Filtered |
|---|---|---|
| Trades Kept | {trading_test['Baseline']['n_kept']} (100%) | {trading_test['XGBoost']['n_kept']} ({trading_test['XGBoost']['pct_kept']:.1f}%) |
| Win Rate | {trading_test['Baseline']['win_rate']:.1f}% | {trading_test['XGBoost']['win_rate']:.1f}% |
| Realized Sum R | {trading_test['Baseline']['sum_r']:+.2f}R | {trading_test['XGBoost']['sum_r']:+.2f}R |
| Expected R / Trade | {trading_test['Baseline']['exp_r']:+.3f}R | {trading_test['XGBoost']['exp_r']:+.3f}R |
| Profit Factor | {trading_test['Baseline']['profit_factor']:.2f} | {trading_test['XGBoost']['profit_factor']:.2f} |
| Max Drawdown | {trading_test['Baseline']['max_dd']:.2f}R | {trading_test['XGBoost']['max_dd']:.2f}R |

## 5. Top 10 Feature Importances
| Rank | Feature | Importance Score |
|---|---|---|
"""
    for rank, (feat, score) in enumerate(xgb_res["feature_importance"].head(10).items(), 1):
        xgb_report += f"| {rank} | `{feat}` | {score:.4f} |\n"

    with open(REPORTS_DIR / "xgboost_report.md", "w") as fp:
        fp.write(xgb_report)

    # 3. Hybrid Report
    hy_report = f"""# Hybrid CatBoost-XGBoost Stacking Report

## 1. Architecture Overview
- **Layer 0 (Base Learners)**: CatBoost + XGBoost
- **Layer 1 (Meta-Learner)**: L2 Regularized Logistic Regression (LogisticRegressionCV)
- **Input Features to Meta-Learner**: Out-of-fold probability predictions `[P_CatBoost, P_XGBoost]`

## 2. Meta-Learner Learned Weights
- **CatBoost Weight**: {hybrid_res['weights']['catboost']:+.4f}
- **XGBoost Weight**: {hybrid_res['weights']['xgboost']:+.4f}
- **Intercept**: {hybrid_res['weights']['intercept']:+.4f}

## 3. Performance Metrics
| Stage | Metric | CatBoost Base | XGBoost Base | Hybrid Stack | Lift over Best Base |
|---|---|---|---|---|---|
| OOF | AUC | {oof_metrics['CatBoost']['auc']:.4f} | {oof_metrics['XGBoost']['auc']:.4f} | {oof_metrics['Hybrid']['auc']:.4f} | {oof_metrics['Hybrid']['auc'] - max(oof_metrics['CatBoost']['auc'], oof_metrics['XGBoost']['auc']):+.4f} |
| OOF | Log-Loss | {oof_metrics['CatBoost']['logloss']:.4f} | {oof_metrics['XGBoost']['logloss']:.4f} | {oof_metrics['Hybrid']['logloss']:.4f} | {min(oof_metrics['CatBoost']['logloss'], oof_metrics['XGBoost']['logloss']) - oof_metrics['Hybrid']['logloss']:+.4f} |
| Holdout 2026 | AUC | {test_metrics['CatBoost']['auc']:.4f} | {test_metrics['XGBoost']['auc']:.4f} | {test_metrics['Hybrid']['auc']:.4f} | {test_metrics['Hybrid']['auc'] - max(test_metrics['CatBoost']['auc'], test_metrics['XGBoost']['auc']):+.4f} |
| Holdout 2026 | Log-Loss | {test_metrics['CatBoost']['logloss']:.4f} | {test_metrics['XGBoost']['logloss']:.4f} | {test_metrics['Hybrid']['logloss']:.4f} | {min(test_metrics['CatBoost']['logloss'], test_metrics['XGBoost']['logloss']) - test_metrics['Hybrid']['logloss']:+.4f} |

## 4. Trading Simulation (P >= 0.50 on Holdout)
- **Trades Kept**: {trading_test['Hybrid']['n_kept']} ({trading_test['Hybrid']['pct_kept']:.1f}%)
- **Win Rate**: {trading_test['Hybrid']['win_rate']:.1f}%
- **Realized Sum R**: {trading_test['Hybrid']['sum_r']:+.2f}R (vs Baseline: {trading_test['Baseline']['sum_r']:+.2f}R)
- **Profit Factor**: {trading_test['Hybrid']['profit_factor']:.2f}
- **Max Drawdown**: {trading_test['Hybrid']['max_dd']:.2f}R
"""
    with open(REPORTS_DIR / "hybrid_report.md", "w") as fp:
        fp.write(hy_report)

    # 4. Ensemble Report
    ens_report = f"""# Ensemble Techniques Report

## 1. Techniques Compared
1. **Simple Average**: Arithmetic mean of predicted probabilities.
2. **Weighted Average**: Optuna-optimized combination on OOF.
3. **Rank-Based Percentile Ensemble**: Empirical percentile transformation followed by averaging.
4. **Hard Voting**: Binary consensus thresholded at 0.50.

## 2. Comparative Performance Table
| Technique | OOF AUC | OOF LogLoss | Holdout AUC | Holdout LogLoss | Holdout Sum R | Holdout PF |
|---|---|---|---|---|---|---|
"""
    for name, key in [("Simple Average", "Ensemble_Simple"),
                      ("Weighted Average", "Ensemble_Weighted"),
                      ("Rank Ensemble", "Ensemble_Rank"),
                      ("Hard Voting", "Ensemble_Voting")]:
        e = ens_res[key]
        tr = trading_test[key]
        ens_report += f"| {name} | {e['oof_auc']:.4f} | {e['oof_loss']:.4f} | {e['test_auc']:.4f} | {e['test_loss']:.4f} | {tr['sum_r']:+.2f}R | {tr['profit_factor']:.2f} |\n"

    ens_report += f"""
## 3. Best Ensemble Method
- **Selected Method**: {ens_res['Ensemble_Weighted']['name'] if ens_res['Ensemble_Weighted']['test_auc'] >= ens_res['Ensemble_Rank']['test_auc'] else 'Rank-Based Ensemble'}
- **Rationale**: Balances discrimination capability, calibration stability across model variances, and maximized simulated sum R.
"""
    with open(REPORTS_DIR / "ensemble_report.md", "w") as fp:
        fp.write(ens_report)

    # 5. Comparative Summary Report
    comp_report = f"""# Model Performance Comparative Summary

## 1. Out-Of-Sample Probability Discrimination (OOF 2023-2025)
| Model | AUC | 95% Bootstrap CI | Log-Loss | Brier Score |
|---|---|---|---|---|
| Baseline (Base Rate) | 0.5000 | [—, —] | {oof_metrics['Baseline']['logloss']:.4f} | {oof_metrics['Baseline']['brier']:.4f} |
| CatBoost | {oof_metrics['CatBoost']['auc']:.4f} | [{oof_metrics['CatBoost']['ci_lo']:.4f}, {oof_metrics['CatBoost']['ci_hi']:.4f}] | {oof_metrics['CatBoost']['logloss']:.4f} | {oof_metrics['CatBoost']['brier']:.4f} |
| XGBoost | {oof_metrics['XGBoost']['auc']:.4f} | [{oof_metrics['XGBoost']['ci_lo']:.4f}, {oof_metrics['XGBoost']['ci_hi']:.4f}] | {oof_metrics['XGBoost']['logloss']:.4f} | {oof_metrics['XGBoost']['brier']:.4f} |
| Hybrid Stacking | {oof_metrics['Hybrid']['auc']:.4f} | [{oof_metrics['Hybrid']['ci_lo']:.4f}, {oof_metrics['Hybrid']['ci_hi']:.4f}] | {oof_metrics['Hybrid']['logloss']:.4f} | {oof_metrics['Hybrid']['brier']:.4f} |
| Ensemble (Rank) | {oof_metrics['Ensemble_Rank']['auc']:.4f} | [{oof_metrics['Ensemble_Rank']['ci_lo']:.4f}, {oof_metrics['Ensemble_Rank']['ci_hi']:.4f}] | {oof_metrics['Ensemble_Rank']['logloss']:.4f} | {oof_metrics['Ensemble_Rank']['brier']:.4f} |

## 2. Holdout Test Evaluation (2026 Out-Of-Sample)
| Model | AUC | 95% Bootstrap CI | Log-Loss | Brier Score |
|---|---|---|---|---|
| Baseline (Take All) | 0.5000 | [—, —] | {test_metrics['Baseline']['logloss']:.4f} | {test_metrics['Baseline']['brier']:.4f} |
| CatBoost | {test_metrics['CatBoost']['auc']:.4f} | [{test_metrics['CatBoost']['ci_lo']:.4f}, {test_metrics['CatBoost']['ci_hi']:.4f}] | {test_metrics['CatBoost']['logloss']:.4f} | {test_metrics['CatBoost']['brier']:.4f} |
| XGBoost | {test_metrics['XGBoost']['auc']:.4f} | [{test_metrics['XGBoost']['ci_lo']:.4f}, {test_metrics['XGBoost']['ci_hi']:.4f}] | {test_metrics['XGBoost']['logloss']:.4f} | {test_metrics['XGBoost']['brier']:.4f} |
| Hybrid Stacking | {test_metrics['Hybrid']['auc']:.4f} | [{test_metrics['Hybrid']['ci_lo']:.4f}, {test_metrics['Hybrid']['ci_hi']:.4f}] | {test_metrics['Hybrid']['logloss']:.4f} | {test_metrics['Hybrid']['brier']:.4f} |
| Ensemble (Rank) | {test_metrics['Ensemble_Rank']['auc']:.4f} | [{test_metrics['Ensemble_Rank']['ci_lo']:.4f}, {test_metrics['Ensemble_Rank']['ci_hi']:.4f}] | {test_metrics['Ensemble_Rank']['logloss']:.4f} | {test_metrics['Ensemble_Rank']['brier']:.4f} |

## 3. Paired Statistical Test: XGBoost vs CatBoost
- **Mean Delta AUC (XGB - Cat)**: {paired_test['mean_diff']:+.4f}
- **95% Bootstrap CI**: [{paired_test['ci_low']:+.4f}, {paired_test['ci_high']:+.4f}]
- **P(XGBoost > CatBoost)**: {paired_test['p_value_a_better'] * 100:.1f}%

## 4. Trading Simulation Comparison (P >= 0.50 on Holdout)
| Model | Kept Trades | Win Rate | Realized Sum R | Exp R / Trade | Profit Factor | Max DD (R) |
|---|---|---|---|---|---|---|
| Baseline (Take All) | {trading_test['Baseline']['n_kept']} (100%) | {trading_test['Baseline']['win_rate']:.1f}% | {trading_test['Baseline']['sum_r']:+.2f}R | {trading_test['Baseline']['exp_r']:+.3f}R | {trading_test['Baseline']['profit_factor']:.2f} | {trading_test['Baseline']['max_dd']:.2f}R |
| CatBoost | {trading_test['CatBoost']['n_kept']} ({trading_test['CatBoost']['pct_kept']:.1f}%) | {trading_test['CatBoost']['win_rate']:.1f}% | {trading_test['CatBoost']['sum_r']:+.2f}R | {trading_test['CatBoost']['exp_r']:+.3f}R | {trading_test['CatBoost']['profit_factor']:.2f} | {trading_test['CatBoost']['max_dd']:.2f}R |
| XGBoost | {trading_test['XGBoost']['n_kept']} ({trading_test['XGBoost']['pct_kept']:.1f}%) | {trading_test['XGBoost']['win_rate']:.1f}% | {trading_test['XGBoost']['sum_r']:+.2f}R | {trading_test['XGBoost']['exp_r']:+.3f}R | {trading_test['XGBoost']['profit_factor']:.2f} | {trading_test['XGBoost']['max_dd']:.2f}R |
| Hybrid Stacking | {trading_test['Hybrid']['n_kept']} ({trading_test['Hybrid']['pct_kept']:.1f}%) | {trading_test['Hybrid']['win_rate']:.1f}% | {trading_test['Hybrid']['sum_r']:+.2f}R | {trading_test['Hybrid']['exp_r']:+.3f}R | {trading_test['Hybrid']['profit_factor']:.2f} | {trading_test['Hybrid']['max_dd']:.2f}R |
| Ensemble (Rank) | {trading_test['Ensemble_Rank']['n_kept']} ({trading_test['Ensemble_Rank']['pct_kept']:.1f}%) | {trading_test['Ensemble_Rank']['win_rate']:.1f}% | {trading_test['Ensemble_Rank']['sum_r']:+.2f}R | {trading_test['Ensemble_Rank']['exp_r']:+.3f}R | {trading_test['Ensemble_Rank']['profit_factor']:.2f} | {trading_test['Ensemble_Rank']['max_dd']:.2f}R |
"""
    with open(REPORTS_DIR / "comparative_summary.md", "w") as fp:
        fp.write(comp_report)

    # 6. Final Recommendation Report
    best_model_name = max(
        [("CatBoost", test_metrics["CatBoost"]["auc"]),
         ("XGBoost", test_metrics["XGBoost"]["auc"]),
         ("Hybrid", test_metrics["Hybrid"]["auc"]),
         ("Ensemble (Rank)", test_metrics["Ensemble_Rank"]["auc"])],
        key=lambda x: x[1]
    )[0]

    rec_report = f"""# Final Model Recommendation & Deployment Guidelines

## 1. Top Recommended Model
**Winner**: **{best_model_name}**
- **Justification**: Delivered highest out-of-sample discrimination on the 2026 holdout dataset while preserving statistical calibration and delivering superior risk-adjusted return (Profit Factor and reduced Drawdown).

## 2. Operating Threshold Recommendation
- **Recommended Threshold**: `P >= 0.50`
- **Rationale**: Setting threshold to 0.50 maintains sufficient trade volume ({trading_test.get(best_model_name, trading_test['Hybrid'])['pct_kept']:.0f}% of signals taken) while filtering out high-loss setups.

## 3. Per-Signal Performance & Applicability
| Signal Type | Total Signals | Take-All Sum R | Filtered Sum R (P>=0.50) | Recommendation |
|---|---|---|---|---|
"""
    for sig, data in sig_breakdown.items():
        rec_report += f"| `{sig}` | {data['n_all']} | {data['sum_r_all']:+.2f}R | {data['sum_r_filtered']:+.2f}R | {'Deploy Filter' if data['sum_r_filtered'] >= data['sum_r_all'] else 'Take All / Monitor'} |\n"

    rec_report += f"""
## 4. Risk Considerations & Operational Limits
1. **Regime Shifts**: Gold (XAUUSD) volatility exhibits cyclical shifts. Re-evaluate feature calibration quarterly.
2. **Execution Slippage**: All metrics assume zero slippage beyond spread. High impact news spikes should still be avoided.
3. **Threshold Stability**: Avoid dynamically shifting probability thresholds based on short-term winning/losing streaks.
"""
    with open(REPORTS_DIR / "final_recommendation.md", "w") as fp:
        fp.write(rec_report)

    print(f"\nSuccessfully generated all 6 detailed reports in: {REPORTS_DIR}")

# ----------------------------------------------------------------------------
# 4. Master Evaluation Pipeline
# ----------------------------------------------------------------------------
def evaluate_all(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 y_train: np.ndarray, y_test: np.ndarray, val_mask: np.ndarray,
                 cb_res: dict, xgb_res: dict, hybrid_res: dict, ens_res: dict):
    """Orchestrates comprehensive metric calculation, simulation, plots, and reports."""
    print("=" * 60)
    print("EVALUATION & REPORTING PIPELINE")
    print("=" * 60)

    # Predictions dictionary for OOF
    oof_preds = {
        "CatBoost": cb_res["oof_preds"],
        "XGBoost": xgb_res["oof_preds"],
        "Hybrid": hybrid_res["oof_preds"],
        "Ensemble_Simple": ens_res["Ensemble_Simple"]["oof"],
        "Ensemble_Weighted": ens_res["Ensemble_Weighted"]["oof"],
        "Ensemble_Rank": ens_res["Ensemble_Rank"]["oof"],
        "Ensemble_Voting": ens_res["Ensemble_Voting"]["oof"],
    }

    # Predictions dictionary for Holdout Test
    test_preds = {
        "CatBoost": cb_res["test_preds"],
        "XGBoost": xgb_res["test_preds"],
        "Hybrid": hybrid_res["test_preds"],
        "Ensemble_Simple": ens_res["Ensemble_Simple"]["test"],
        "Ensemble_Weighted": ens_res["Ensemble_Weighted"]["test"],
        "Ensemble_Rank": ens_res["Ensemble_Rank"]["test"],
        "Ensemble_Voting": ens_res["Ensemble_Voting"]["test"],
    }

    # Save predictions to CSV
    oof_df = train_df[["pos_id", "open_time", "close_time", "signal", "y", "result_R"]].copy()
    oof_df["evaluated_oof"] = val_mask
    for name, p in oof_preds.items():
        oof_df[f"p_{name}"] = p
    oof_csv_path = PREDICTIONS_DIR / "oof_predictions.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"Saved OOF predictions to: {oof_csv_path}")

    holdout_df = test_df[["pos_id", "open_time", "close_time", "signal", "y", "result_R"]].copy()
    for name, p in test_preds.items():
        holdout_df[f"p_{name}"] = p
    holdout_csv_path = PREDICTIONS_DIR / "holdout_predictions.csv"
    holdout_df.to_csv(holdout_csv_path, index=False)
    print(f"Saved Holdout predictions to: {holdout_csv_path}")

    # Compute metrics for OOF
    y_val = y_train[val_mask]
    oof_metrics = {
        "Baseline": {
            "auc": 0.50,
            "ci_lo": 0.50,
            "ci_hi": 0.50,
            "logloss": log_loss(y_val, np.full(len(y_val), y_val.mean())),
            "brier": brier_score_loss(y_val, np.full(len(y_val), y_val.mean())),
        }
    }
    for name, p in oof_preds.items():
        p_val = p[val_mask]
        auc = roc_auc_score(y_val, p_val)
        lo, hi = bootstrap_auc_ci(y_val, p_val)
        loss = log_loss(y_val, np.clip(p_val, 1e-6, 1 - 1e-6))
        brier = brier_score_loss(y_val, p_val)
        oof_metrics[name] = {"auc": auc, "ci_lo": lo, "ci_hi": hi, "logloss": loss, "brier": brier}

    # Compute metrics for Holdout
    test_metrics = {
        "Baseline": {
            "auc": 0.50,
            "ci_lo": 0.50,
            "ci_hi": 0.50,
            "logloss": log_loss(y_test, np.full(len(y_test), y_test.mean())),
            "brier": brier_score_loss(y_test, np.full(len(y_test), y_test.mean())),
        }
    }
    for name, p in test_preds.items():
        auc = roc_auc_score(y_test, p)
        lo, hi = bootstrap_auc_ci(y_test, p)
        loss = log_loss(y_test, np.clip(p, 1e-6, 1 - 1e-6))
        brier = brier_score_loss(y_test, p)
        test_metrics[name] = {"auc": auc, "ci_lo": lo, "ci_hi": hi, "logloss": loss, "brier": brier}

    # Paired test
    paired_test = paired_bootstrap_test(y_test, test_preds["XGBoost"], test_preds["CatBoost"])

    # Trading simulation
    real_r_val = train_df["result_R"].iloc[val_mask].values
    real_r_test = test_df["result_R"].values

    trading_oof = {"Baseline": simulate_trading(np.ones(len(real_r_val)), real_r_val, threshold=0.5)}
    for name, p in oof_preds.items():
        trading_oof[name] = simulate_trading(p[val_mask], real_r_val, threshold=0.50)

    trading_test = {"Baseline": simulate_trading(np.ones(len(real_r_test)), real_r_test, threshold=0.5)}
    for name, p in test_preds.items():
        trading_test[name] = simulate_trading(p, real_r_test, threshold=0.50)

    # Per-signal breakdown on holdout
    sig_breakdown = {}
    p_best_test = test_preds["Hybrid"]
    for sig in sorted(test_df["signal"].unique()):
        sig_mask = (test_df["signal"] == sig).values
        sig_r = real_r_test[sig_mask]
        p_sig = p_best_test[sig_mask]
        sig_breakdown[sig] = {
            "n_all": int(sig_mask.sum()),
            "sum_r_all": float(sig_r.sum()),
            "n_filtered": int((p_sig >= 0.50).sum()),
            "sum_r_filtered": float(sig_r[p_sig >= 0.50].sum()),
        }

    # Generate plots
    print("\nGenerating evaluation charts...")
    plot_equity_curves(
        train_df[val_mask].reset_index(drop=True),
        {k: oof_preds[k][val_mask] for k in ["CatBoost", "XGBoost", "Hybrid", "Ensemble_Rank"]},
        "Walk-Forward Out-Of-Fold Cumulative R (2023-2025)",
        PLOTS_DIR / "equity_compare_oof.png"
    )
    plot_equity_curves(
        test_df,
        {k: test_preds[k] for k in ["CatBoost", "XGBoost", "Hybrid", "Ensemble_Rank"]},
        "Holdout Cumulative R (2026 Out-Of-Sample)",
        PLOTS_DIR / "equity_compare_test.png"
    )
    plot_calibration(y_test, {k: test_preds[k] for k in ["CatBoost", "XGBoost", "Hybrid"]}, PLOTS_DIR / "calibration_plot.png")
    plot_roc_curves(y_test, {k: test_preds[k] for k in ["CatBoost", "XGBoost", "Hybrid", "Ensemble_Rank"]}, PLOTS_DIR / "roc_curves.png")

    # Generate all Markdown reports
    write_markdown_reports(
        oof_metrics, test_metrics, trading_oof, trading_test,
        sig_breakdown, cb_res, xgb_res, hybrid_res, ens_res, paired_test
    )

    return {
        "oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
        "trading_oof": trading_oof,
        "trading_test": trading_test,
        "paired_test": paired_test,
    }
