# feature_engineering/pipeline.py

import json
import os
from typing import Dict, Any, Optional

import pandas as pd

from .config import FeatureConfig, LSTDExportConfig
from .engineering import apply_feature_pipeline
from .export import build_lstd_ready_frame
from datasets.utils import ensure_dir


def engineer_historical_features(
    raw_df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    export_cfg: Optional[LSTDExportConfig] = None,
    output_dir: Optional[str] = None,
    output_stem: Optional[str] = None,
) -> Dict[str, Any]:
    feat_df = apply_feature_pipeline(raw_df, feature_cfg)

    result: Dict[str, Any] = {
        "feature_rows": len(feat_df),
        "feature_columns": list(feat_df.columns),
        "features_df": feat_df,
    }

    if output_dir is not None and output_stem is not None:
        ensure_dir(output_dir)
        feat_path = os.path.join(output_dir, f"{output_stem}_features.csv")
        feat_df.to_csv(feat_path, index=False)
        result["features_path"] = feat_path

    if export_cfg is not None and export_cfg.enabled:
        lstd_df, used_feature_cols = build_lstd_ready_frame(feat_df, export_cfg)
        result["lstd_df"] = lstd_df
        result["used_feature_columns"] = used_feature_cols

        if output_dir is not None and output_stem is not None:
            lstd_path = os.path.join(output_dir, f"{output_stem}{export_cfg.lstd_csv_suffix}")
            meta_path = os.path.join(output_dir, f"{output_stem}{export_cfg.metadata_suffix}")
            lstd_df.to_csv(lstd_path, index=False)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "used_feature_columns": used_feature_cols,
                        "feature_mode": export_cfg.feature_mode,
                        "target_column": export_cfg.target_column,
                        "lstd_csv_path": lstd_path,
                    },
                    f,
                    indent=2,
                )
            result["lstd_csv_path"] = lstd_path
            result["metadata_path"] = meta_path

    return result