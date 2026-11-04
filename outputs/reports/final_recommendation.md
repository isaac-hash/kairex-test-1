# Final Model Recommendation & Deployment Guidelines

## 1. Top Recommended Model
**Winner**: **XGBoost**
- **Justification**: Delivered highest out-of-sample discrimination on the 2026 holdout dataset while preserving statistical calibration and delivering superior risk-adjusted return (Profit Factor and reduced Drawdown).

## 2. Operating Threshold Recommendation
- **Recommended Threshold**: `P >= 0.50`
- **Rationale**: Setting threshold to 0.50 maintains sufficient trade volume (49% of signals taken) while filtering out high-loss setups.

## 3. Per-Signal Performance & Applicability
| Signal Type | Total Signals | Take-All Sum R | Filtered Sum R (P>=0.50) | Recommendation |
|---|---|---|---|---|
| `SIG_BREAKER_M30` | 14 | +17.98R | +17.98R | Deploy Filter |
| `SIG_FAILEDENG_M30` | 36 | +19.89R | +23.91R | Deploy Filter |
| `SIG_TYPE4_M15_M3` | 181 | +38.65R | +16.10R | Take All / Monitor |
| `SIG_TYPE4_M30` | 98 | +0.91R | +0.98R | Deploy Filter |
| `SIG_TYPE5_M15_M5` | 173 | -19.42R | -8.86R | Deploy Filter |
| `SIG_TYPE5_M30` | 97 | +8.51R | +8.71R | Deploy Filter |

## 4. Risk Considerations & Operational Limits
1. **Regime Shifts**: Gold (XAUUSD) volatility exhibits cyclical shifts. Re-evaluate feature calibration quarterly.
2. **Execution Slippage**: All metrics assume zero slippage beyond spread. High impact news spikes should still be avoided.
3. **Threshold Stability**: Avoid dynamically shifting probability thresholds based on short-term winning/losing streaks.
