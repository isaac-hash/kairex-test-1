# Hybrid CatBoost-XGBoost Stacking Report

## 1. Architecture Overview
- **Layer 0 (Base Learners)**: CatBoost + XGBoost
- **Layer 1 (Meta-Learner)**: L2 Regularized Logistic Regression (LogisticRegressionCV)
- **Input Features to Meta-Learner**: Out-of-fold probability predictions `[P_CatBoost, P_XGBoost]`

## 2. Meta-Learner Learned Weights
- **CatBoost Weight**: +2.5800
- **XGBoost Weight**: +2.1244
- **Intercept**: -2.2464

## 3. Performance Metrics
| Stage | Metric | CatBoost Base | XGBoost Base | Hybrid Stack | Lift over Best Base |
|---|---|---|---|---|---|
| OOF | AUC | 0.6103 | 0.5826 | 0.6155 | +0.0052 |
| OOF | Log-Loss | 0.6783 | 0.6806 | 0.6746 | +0.0036 |
| Holdout 2026 | AUC | 0.4741 | 0.4818 | 0.4764 | -0.0055 |
| Holdout 2026 | Log-Loss | 0.7486 | 0.7576 | 0.7533 | -0.0047 |

## 4. Trading Simulation (P >= 0.50 on Holdout)
- **Trades Kept**: 402 (67.1%)
- **Win Rate**: 25.6%
- **Realized Sum R**: +58.82R (vs Baseline: +66.52R)
- **Profit Factor**: 1.19
- **Max Drawdown**: 44.38R
