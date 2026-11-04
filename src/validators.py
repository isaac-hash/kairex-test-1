"""
Validation strategy module for Kairex PSS SCALP ML Pipeline.
Implements purged expanding-window walk-forward cross validation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Generator, Tuple, Dict, Any

class PurgedWalkForwardCV:
    """
    Purged Expanding-Window Walk-Forward Cross-Validator.
    
    Guarantees:
    1. Chronological order is strictly preserved.
    2. Expanding training window.
    3. Purging: Only trades that had completely closed before the test window opened
       are included in the training fold.
    """
    def __init__(self, n_folds: int = 5, initial_train_pct: float = 0.50):
        self.n_folds = n_folds
        self.initial_train_pct = initial_train_pct

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray, Dict[str, Any]], None, None]:
        """
        Yields (train_indices, val_indices, fold_metadata).
        df must contain 'open_dt' and 'close_dt' and be sorted by open_dt.
        """
        n_samples = len(df)
        start_idx = int(n_samples * self.initial_train_pct)
        edges = np.linspace(start_idx, n_samples, self.n_folds + 1).astype(int)

        for fold in range(self.n_folds):
            val_idx = np.arange(edges[fold], edges[fold + 1])
            val_start_dt = df["open_dt"].iloc[val_idx[0]]
            val_end_dt = df["open_dt"].iloc[val_idx[-1]]

            # Purging: Only trades closed strictly prior to val_start_dt
            # and prior to the fold boundary
            candidate_train = np.where(df["close_dt"] < val_start_dt)[0]
            train_idx = candidate_train[candidate_train < edges[fold]]

            fold_info = {
                "fold": fold + 1,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "val_start_dt": str(val_start_dt),
                "val_end_dt": str(val_end_dt),
                "train_base_rate": float(df["y"].iloc[train_idx].mean()),
                "val_base_rate": float(df["y"].iloc[val_idx].mean()),
            }

            yield train_idx, val_idx, fold_info

if __name__ == "__main__":
    from src.data_loader import prepare_datasets
    train_df, _, _, _, _, _, _ = prepare_datasets()
    cv = PurgedWalkForwardCV(n_folds=5)
    for tr, val, info in cv.split(train_df):
        print(f"Fold {info['fold']}: Train N={info['n_train']} -> Val N={info['n_val']} | {info['val_start_dt']} to {info['val_end_dt']}")
