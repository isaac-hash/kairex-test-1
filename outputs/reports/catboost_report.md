# CatBoost Model Performance Report

## 1. Optimal Hyperparameters
```json
{
  "iterations":200,
  "depth":6,
  "learning_rate":0.0349028546,
  "l2_leaf_reg":11.2049435113,
  "random_strength":0.1391105392,
  "eval_metric":"Logloss",
  "bagging_temperature":0.9699098522
}
```

## 2. Walk-Forward Cross-Validation (Out-Of-Fold)
- **Aggregated OOF AUC**: 0.6103 (95% CI: [0.5591, 0.6600])
- **Log-Loss**: 0.6783
- **Brier Score**: 0.2427

### Per-Fold Breakdown:
| Fold | Train N | Val N | Val AUC | Val LogLoss |
|---|---|---|---|---|
| 1 | 472 | 95 | 0.6986 | 0.6539 |
| 2 | 570 | 95 | 0.5501 | 0.6858 |
| 3 | 665 | 95 | 0.6320 | 0.6677 |
| 4 | 758 | 95 | 0.5386 | 0.6930 |
| 5 | 855 | 96 | 0.5798 | 0.6909 |

## 3. Holdout Test Set (2026) Evaluation
- **Holdout AUC**: 0.4741 (95% CI: [0.4283, 0.5192])
- **Holdout Log-Loss**: 0.7486
- **Holdout Brier Score**: 0.2755

## 4. Trading Simulation (Decision Rule: Execute if P >= 0.50)
| Metric | Take-All Baseline | CatBoost Filtered |
|---|---|---|
| Trades Kept | 599 (100%) | 426 (71.1%) |
| Win Rate | 25.7% | 23.7% |
| Realized Sum R | +66.52R | +25.98R |
| Expected R / Trade | +0.111R | +0.061R |
| Profit Factor | 1.15 | 1.08 |
| Max Drawdown | 55.08R | 45.43R |

## 5. Top 10 Feature Importances
| Rank | Feature | Importance Score |
|---|---|---|
| 1 | `day_range_to_adr` | 7.080 |
| 2 | `dist_ema200_h4_atr` | 4.820 |
| 3 | `dist_round_R` | 3.395 |
| 4 | `day_of_month` | 2.995 |
| 5 | `prev_R_3` | 2.991 |
| 6 | `mins_since_ny` | 2.932 |
| 7 | `sl_to_atr_m15` | 2.891 |
| 8 | `atr_ratio_h1h4` | 2.833 |
| 9 | `mom5_atr` | 2.777 |
| 10 | `rsi_diff` | 2.727 |
