# Model Performance Comparative Summary

## 1. Out-Of-Sample Probability Discrimination (OOF 2023-2025)
| Model | AUC | 95% Bootstrap CI | Log-Loss | Brier Score |
|---|---|---|---|---|
| Baseline (Base Rate) | 0.5000 | [—, —] | 0.6924 | 0.2496 |
| CatBoost | 0.6103 | [0.5591, 0.6600] | 0.6783 | 0.2427 |
| XGBoost | 0.5826 | [0.5312, 0.6330] | 0.6806 | 0.2438 |
| Hybrid Stacking | 0.6155 | [0.5614, 0.6649] | 0.6746 | 0.2409 |
| Ensemble (Rank) | 0.6098 | [0.5576, 0.6589] | 0.7616 | 0.2628 |

## 2. Holdout Test Evaluation (2026 Out-Of-Sample)
| Model | AUC | 95% Bootstrap CI | Log-Loss | Brier Score |
|---|---|---|---|---|
| Baseline (Take All) | 0.5000 | [—, —] | 0.6931 | 0.2500 |
| CatBoost | 0.4741 | [0.4283, 0.5192] | 0.7486 | 0.2755 |
| XGBoost | 0.4818 | [0.4374, 0.5291] | 0.7576 | 0.2783 |
| Hybrid Stacking | 0.4764 | [0.4311, 0.5216] | 0.7533 | 0.2773 |
| Ensemble (Rank) | 0.4755 | [0.4304, 0.5207] | 0.9367 | 0.3266 |

## 3. Paired Statistical Test: XGBoost vs CatBoost
- **Mean Delta AUC (XGB - Cat)**: +0.0086
- **95% Bootstrap CI**: [-0.0340, +0.0504]
- **P(XGBoost > CatBoost)**: 64.7%

## 4. Trading Simulation Comparison (P >= 0.50 on Holdout)
| Model | Kept Trades | Win Rate | Realized Sum R | Exp R / Trade | Profit Factor | Max DD (R) |
|---|---|---|---|---|---|---|
| Baseline (Take All) | 599 (100%) | 25.7% | +66.52R | +0.111R | 1.15 | 55.08R |
| CatBoost | 426 (71.1%) | 23.7% | +25.98R | +0.061R | 1.08 | 45.43R |
| XGBoost | 294 (49.1%) | 24.5% | +26.58R | +0.090R | 1.12 | 44.52R |
| Hybrid Stacking | 402 (67.1%) | 25.6% | +58.82R | +0.146R | 1.19 | 44.38R |
| Ensemble (Rank) | 302 (50.4%) | 23.8% | +26.64R | +0.088R | 1.11 | 43.40R |
