# feature_engineering/export.py

from typing import List

import pandas as pd

from .config import LSTDExportConfig


def build_lstd_ready_frame(df: pd.DataFrame, export_cfg: LSTDExportConfig) -> tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    if "open_dt" not in out.columns:
        raise ValueError("Expected 'open_dt' column in engineered dataframe.")

    out["date"] = pd.to_datetime(out["open_dt"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    numeric_cols = list(out.select_dtypes(include=["number"]).columns)

    exclude = set(export_cfg.exclude_columns)
    exclude.add("open_dt")
    exclude.add("close_dt")

    if export_cfg.drop_explicit_target_columns:
        for c in list(numeric_cols):
            if c.startswith("target_"):
                exclude.add(c)

    if export_cfg.target_column not in out.columns:
        raise ValueError(f"target_column '{export_cfg.target_column}' not found in dataframe")

    if export_cfg.feature_columns is None:
        feature_cols = [c for c in numeric_cols if c not in exclude]
    else:
        missing = [c for c in export_cfg.feature_columns if c not in out.columns]
        if missing:
            raise ValueError(f"feature_columns missing from dataframe: {missing}")
        feature_cols = [c for c in export_cfg.feature_columns if c not in exclude]

    if export_cfg.target_column in feature_cols:
        feature_cols.remove(export_cfg.target_column)

    mode = export_cfg.feature_mode.upper()
    if mode == "S":
        final_cols = ["date", export_cfg.target_column]
        used_feature_cols = [export_cfg.target_column]
    elif mode in {"M", "MS"}:
        final_cols = ["date"] + feature_cols + [export_cfg.target_column]
        used_feature_cols = feature_cols + [export_cfg.target_column]
    else:
        raise ValueError("feature_mode must be one of: S, M, MS")

    lstd_df = out[final_cols].copy()
    return lstd_df, used_feature_cols