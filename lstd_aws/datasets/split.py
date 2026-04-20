# datasets/split.py

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ChronologicalSplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    test_warmup_bars: int = 0


@dataclass
class RawSplitResult:
    train_raw: pd.DataFrame
    val_raw: pd.DataFrame
    test_raw_with_warmup: pd.DataFrame
    test_effective_start_open_time: Optional[int]



def split_raw_for_train_test(raw_df: pd.DataFrame, cfg: ChronologicalSplitConfig) -> RawSplitResult:
    if abs((cfg.train_ratio + cfg.val_ratio + cfg.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n = len(raw_df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)

    train_raw = raw_df.iloc[:n_train].copy()
    val_raw = raw_df.iloc[n_train:n_train + n_val].copy()

    test_start = n_train + n_val
    warmup_start = max(0, test_start - cfg.test_warmup_bars)
    test_raw_with_warmup = raw_df.iloc[warmup_start:].copy()

    effective_open_time = None
    if test_start < len(raw_df):
        effective_open_time = int(raw_df.iloc[test_start]["open_time"])

    return RawSplitResult(
        train_raw=train_raw,
        val_raw=val_raw,
        test_raw_with_warmup=test_raw_with_warmup,
        test_effective_start_open_time=effective_open_time,
    )