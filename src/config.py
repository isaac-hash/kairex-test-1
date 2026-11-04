"""
Configuration and constants for the Kairex PSS SCALP ML Pipeline.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
TRAIN_CSV = DATA_DIR / "PSS_SCALP_XAUUSDm 2023-2025.csv"
TEST_CSV = DATA_DIR / "PSS_SCALP_XAUUSDm 2026 new.csv"

OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure output directories exist
for d in [OUTPUTS_DIR, MODELS_DIR, PREDICTIONS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random seed
SEED = 42

# Columns that represent post-entry data leakage and must never be in X
LEAKAGE_COLS = [
    "pos_id",
    "pair",
    "date",
    "open_time",
    "close_time",
    "duration_min",
    "entry",
    "sl",
    "tp",
    "close_price",
    "lots",
    "risk_money",
    "net_profit",
    "result_R",
    "pnl_R",
    "exit_reason",
    "MFE_R",
    "MAE_R",
    "max_p_b4_sl",
    "max_L_b4_tp",
    "excursion_flag",
    "min_to_MFE",
    "min_to_MAE",
    "min_to_1R",
    "min_to_2R",
    "min_to_3R",
    "min_R_after_1R",
    "min_R_after_2R",
    "dist_opp_zone_R",  # 100% NaN in both datasets
    "direction",        # Constant ('BUY') across all trades
]

# Base categorical columns
CAT_FEATURES = [
    "signal",
    "signal_type",
    "zone_timeframe",
    "entry_timeframe",
    "entry_day",
]

# Numerical candidate features known strictly at trade entry
NUM_FEATURES = [
    "sl_pips",
    "tp_pips",
    "spread_pips",
    "risk_pts",
    "spread_pts",
    "planned_rr",
    "zone_age",
    "zone_size_pips",
    "zone_touches",
    "body_size_ratio",
    "engulf_depth",
    "atr_m5_pips",
    "atr_m15_pips",
    "atr_h1_pips",
    "atr_h4_pips",
    "sl_to_atr_m15",
    "sl_to_atr_h1",
    "zone_to_atr_m15",
    "atr_h1_pctile",
    "day_range_to_adr",
    "dist_ema50_h1_atr",
    "dist_ema200_h1_atr",
    "dist_ema50_h4_atr",
    "dist_ema200_h4_atr",
    "slope_ema50_h1_atr",
    "slope_ema50_h4_atr",
    "align_ema50_h1",
    "align_ema50_h4",
    "dist_pdh_R",
    "dist_pdl_R",
    "dist_dayhigh_R",
    "dist_daylow_R",
    "dist_round_R",
    "ec_range_atr",
    "ec_close_loc",
    "ec_upper_wick",
    "ec_lower_wick",
    "ec_tickvol_ratio",
    "rsi_m5",
    "rsi_m15",
    "mom5_atr",
    "consec_dir_bars",
    "mins_since_london",
    "mins_since_ny",
    "open_hour",
    "day_of_month",
    "server_gmt_offset_h",
    "prev_R_1",
    "prev_R_2",
    "prev_R_3",
    "trades_today_before",
    "open_pos_other",
    "mins_since_last_entry",
]

# Derived features generated in features.py
DERIVED_FEATURES = [
    "spread_to_sl",
    "zone_to_sl",
    "engulf_to_sl",
    "log_zone_age",
    "log_sl_pips",
    "log_body",
    "atr_ratio_h1h4",
    "rsi_diff",
    "session_tag",
    "prev_R_streak",
]

# Decision threshold
DECISION_THRESHOLD = 0.50
SWEEP_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
