# XGBoost Model Performance Report

## 1. Optimal Hyperparameters
```json
{
  "n_estimators":250.0,
  "max_depth":5.0,
  "learning_rate":0.0493038319,
  "subsample":0.7419018674,
  "colsample_bytree":0.8054653428,
  "min_child_weight":4.0,
  "reg_lambda":2.233504383,
  "reg_alpha":2.5128573223
}
```

## 2. Walk-Forward Cross-Validation (Out-Of-Fold)
- **Aggregated OOF AUC**: 0.5826 (95% CI: [0.5312, 0.6330])
- **Log-Loss**: 0.6806
- **Brier Score**: 0.2438

### Per-Fold Breakdown:
| Fold | Train N | Val N | Val AUC | Val LogLoss |
|---|---|---|---|---|
| 1 | 472 | 95 | 0.6662 | 0.6610 |
| 2 | 570 | 95 | 0.4696 | 0.6953 |
| 3 | 665 | 95 | 0.6459 | 0.6715 |
| 4 | 758 | 95 | 0.5346 | 0.6937 |
| 5 | 855 | 96 | 0.5740 | 0.6812 |

## 3. Holdout Test Set (2026) Evaluation
- **Holdout AUC**: 0.4818 (95% CI: [0.4374, 0.5291])
- **Holdout Log-Loss**: 0.7576
- **Holdout Brier Score**: 0.2783

## 4. Trading Simulation (Decision Rule: Execute if P >= 0.50)
| Metric | Take-All Baseline | XGBoost Filtered |
|---|---|---|
| Trades Kept | 599 (100%) | 294 (49.1%) |
| Win Rate | 25.7% | 24.5% |
| Realized Sum R | +66.52R | +26.58R |
| Expected R / Trade | +0.111R | +0.090R |
| Profit Factor | 1.15 | 1.12 |
| Max Drawdown | 55.08R | 44.52R |

## 5. Top 10 Feature Importances
| Rank | Feature | Importance Score |
|---|---|---|
| 1 | `spread_pts` | 0.0239 |
| 2 | `day_range_to_adr` | 0.0228 |
| 3 | `log_sl_pips` | 0.0220 |
| 4 | `open_pos_other` | 0.0212 |
| 5 | `atr_m15_pips` | 0.0206 |
| 6 | `sl_to_atr_m15` | 0.0199 |
| 7 | `day_of_month` | 0.0198 |
| 8 | `prev_R_3` | 0.0198 |
| 9 | `dist_ema200_h4_atr` | 0.0192 |
| 10 | `entry_day` | 0.0187 |
