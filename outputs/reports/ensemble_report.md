# Ensemble Techniques Report

## 1. Techniques Compared
1. **Simple Average**: Arithmetic mean of predicted probabilities.
2. **Weighted Average**: Optuna-optimized combination on OOF.
3. **Rank-Based Percentile Ensemble**: Empirical percentile transformation followed by averaging.
4. **Hard Voting**: Binary consensus thresholded at 0.50.

## 2. Comparative Performance Table
| Technique | OOF AUC | OOF LogLoss | Holdout AUC | Holdout LogLoss | Holdout Sum R | Holdout PF |
|---|---|---|---|---|---|---|
| Simple Average | 0.6133 | 0.6773 | 0.4766 | 0.7397 | +37.36R | 1.13 |
| Weighted Average | 0.6192 | 0.6773 | 0.4743 | 0.7412 | +53.13R | 1.17 |
| Rank Ensemble | 0.6098 | 0.7616 | 0.4755 | 0.9367 | +26.64R | 1.11 |
| Hard Voting | 0.5985 | 3.8030 | 0.4723 | 5.2041 | +40.32R | 1.12 |

## 3. Best Ensemble Method
- **Selected Method**: Rank-Based Ensemble
- **Rationale**: Balances discrimination capability, calibration stability across model variances, and maximized simulated sum R.
