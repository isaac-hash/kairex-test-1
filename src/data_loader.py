"""
Data loading and preprocessing module for Kairex PSS SCALP ML Pipeline.
Enforces strict anti-leakage guidelines, train-fitted median imputation,
and test set deduplication.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.config import (
    TRAIN_CSV,
    TEST_CSV,
    LEAKAGE_COLS,
    CAT_FEATURES,
    NUM_FEATURES,
    DERIVED_FEATURES,
)
from src.features import engineer_features, get_feature_names

def load_raw_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Loads CSV, parses datetimes, removes open trades, and sorts chronologically."""
    df = pd.read_csv(csv_path)
    
    # Remove unfinished trades with no resolved label
    df = df[df["exit_reason"] != "OPEN"].copy()
    
    # Parse datetimes
    df["open_dt"] = pd.to_datetime(df["open_time"], format="%Y.%m.%d %H:%M:%S")
    df["close_dt"] = pd.to_datetime(df["close_time"], format="%Y.%m.%d %H:%M:%S")
    
    df = df.sort_values("open_dt").reset_index(drop=True)
    
    # Label: 1 if trade touched +1R before stopping out, 0 otherwise
    df["y"] = df["min_to_1R"].notna().astype(int)
    
    return df

def prepare_datasets(dedup_test: bool = True):
    """
    Loads train and holdout test datasets, engineers features, performs
    leakage checks, and imputes missing values strictly using training stats.
    
    Returns:
        train_df: Full train dataframe with metadata, labels, and features
        test_df: Full holdout test dataframe with metadata, labels, and features
        X_train: Feature matrix for training
        y_train: Binary labels for training
        X_test: Feature matrix for testing
        y_test: Binary labels for testing
        feature_info: dict with list of categorical and numerical features
    """
    print(f"Loading training data from: {TRAIN_CSV.name}")
    train_df = load_raw_dataset(TRAIN_CSV)
    
    print(f"Loading holdout test data from: {TEST_CSV.name}")
    test_df = load_raw_dataset(TEST_CSV)
    
    if dedup_test:
        # Detect and remove trades in test that overlap with train
        train_open_times = set(train_df["open_time"])
        overlap_mask = test_df["open_time"].isin(train_open_times)
        num_overlap = overlap_mask.sum()
        if num_overlap > 0:
            print(f"[Purge] Purging {num_overlap} duplicate trades from test set that already exist in train.")
            test_df = test_df[~overlap_mask].reset_index(drop=True)
    
    # Engineer entry-time features
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    cat_cols, num_cols, all_feats = get_feature_names()
    
    # Anti-leakage assertion
    for col in all_feats:
        if col in LEAKAGE_COLS:
            raise ValueError(f"LEAKAGE ALERT: Feature '{col}' is in the forbidden leakage list!")
    
    # Imputation: Compute medians on TRAIN ONLY, apply to both
    train_medians = train_df[num_cols].median()
    
    X_train = train_df[all_feats].copy()
    X_test = test_df[all_feats].copy()
    
    X_train[num_cols] = X_train[num_cols].fillna(train_medians)
    X_test[num_cols] = X_test[num_cols].fillna(train_medians)
    
    for c in cat_cols:
        X_train[c] = X_train[c].astype(str).fillna("MISSING")
        X_test[c] = X_test[c].astype(str).fillna("MISSING")
        
    y_train = train_df["y"].values
    y_test = test_df["y"].values
    
    print(f"Dataset prepared successfully:")
    print(f"  Train: {len(X_train)} samples, Base rate P(+1R): {y_train.mean():.4f}")
    print(f"  Test:  {len(X_test)} samples, Base rate P(+1R): {y_test.mean():.4f}")
    print(f"  Total features: {len(all_feats)} ({len(cat_cols)} categorical, {len(num_cols)} numerical)")
    
    feature_info = {
        "cat_features": cat_cols,
        "num_features": num_cols,
        "all_features": all_feats,
        "train_medians": train_medians.to_dict(),
    }
    
    return train_df, test_df, X_train, y_train, X_test, y_test, feature_info

if __name__ == "__main__":
    tr_df, te_df, X_tr, y_tr, X_te, y_te, feats = prepare_datasets()
    print("Columns in X_tr:", list(X_tr.columns))
