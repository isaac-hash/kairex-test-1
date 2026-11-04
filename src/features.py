"""
Feature engineering for Kairex PSS SCALP ML Pipeline.
Strictly entry-time only: computes ratios and transformations without lookahead.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.config import CAT_FEATURES, NUM_FEATURES, DERIVED_FEATURES

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific ratios and non-linear transformations
    strictly knowable at trade entry.
    """
    df = df.copy()

    # Ratios against SL distance
    df["spread_to_sl"] = df["spread_pips"] / (df["sl_pips"] + 1e-6)
    df["zone_to_sl"] = df["zone_size_pips"] / (df["sl_pips"] + 1e-6)
    df["engulf_to_sl"] = df["engulf_depth"] / (df["sl_pips"] + 1e-6)

    # Log transforms for skewed quantities
    df["log_zone_age"] = np.log1p(np.maximum(0, df["zone_age"].fillna(0)))
    df["log_sl_pips"] = np.log1p(np.maximum(0, df["sl_pips"].fillna(0)))
    df["log_body"] = np.log1p(np.maximum(0, df["body_size_ratio"].fillna(0)))

    # Multi-timeframe volatility ratio
    df["atr_ratio_h1h4"] = df["atr_h1_pips"] / (df["atr_h4_pips"] + 1e-6)

    # Momentum divergence
    df["rsi_diff"] = df["rsi_m15"] - df["rsi_m5"]

    # Market session indicator
    # 2 if early London (< 2h), 1 if early NY (< 2h), 3 if both, 0 otherwise
    is_london = (df["mins_since_london"] >= 0) & (df["mins_since_london"] < 120)
    is_ny = (df["mins_since_ny"] >= 0) & (df["mins_since_ny"] < 120)
    df["session_tag"] = (is_london.astype(int) * 2 + is_ny.astype(int))

    # Streak of losses in previous 3 trades
    r1 = df["prev_R_1"].fillna(0)
    r2 = df["prev_R_2"].fillna(0)
    r3 = df["prev_R_3"].fillna(0)
    df["prev_R_streak"] = ((r1 < 0) & (r2 < 0) & (r3 < 0)).astype(int)

    return df

def get_feature_names():
    """
    Returns full list of features: (categorical, numerical, all_features)
    """
    all_num = NUM_FEATURES + DERIVED_FEATURES
    all_cat = CAT_FEATURES
    return all_cat, all_num, all_cat + all_num
