# feature_engineering/config.py

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FeatureConfig:
    add_basic_price_features: bool = True

    add_instance_norm_features: bool = False
    instance_norm_window: int = 96

    add_seasonal_trend_features: bool = True
    seasonal_trend_window: int = 96

    add_frequency_features: bool = True
    frequency_window: int = 128

    add_long_short_regime_features: bool = True
    short_window: int = 16
    long_window: int = 96

    add_explicit_targets: bool = False
    target_horizon_bars: int = 1

    drop_na_rows: bool = True


@dataclass
class LSTDExportConfig:
    enabled: bool = True
    feature_mode: str = "M"  # S, M, MS
    target_column: str = "close"
    feature_columns: Optional[List[str]] = None
    exclude_columns: List[str] = field(default_factory=lambda: [
        "ignore",
        "open_time",
        "close_time",
    ])
    drop_explicit_target_columns: bool = True
    lstd_csv_suffix: str = "_lstd.csv"
    metadata_suffix: str = "_meta.json"