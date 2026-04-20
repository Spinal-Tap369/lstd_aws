# orchestration/run.py

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

from binanace_history.pipeline import download_historical_klines
from live_inference.service import LSTDLiveInferenceService
from training.trainer import LSTDFitTrainer
from training.utils import ensure_dir, save_json

from .config import PipelineConfig


def write_default_config(path: str) -> str:
    cfg = PipelineConfig.default_btc_1m_real()
    out_dir = os.path.dirname(path)
    if out_dir:
        ensure_dir(out_dir)
    cfg.save_json(path)
    return path


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    cfg = deepcopy(cfg)

    download_result: Optional[Dict[str, Any]] = None
    fit_summary: Optional[Dict[str, Any]] = None
    live_summary: Optional[Dict[str, Any]] = None

    # -------------------------------------------------
    # Step 1: download raw data
    # -------------------------------------------------
    if cfg.execution.run_download:
        download_result = download_historical_klines(cfg.download)
        raw_path = download_result["raw_path"]
        cfg.train.root_path = os.path.dirname(raw_path)
        cfg.train.data_path = os.path.basename(raw_path)
    else:
        if not cfg.train.data_path:
            raise ValueError(
                "run_download is False, so train.root_path/train.data_path must point to an existing raw csv."
            )

    # -------------------------------------------------
    # Step 2: offline fit + artifact bundle publish
    # -------------------------------------------------
    if cfg.execution.run_train:
        trainer = LSTDFitTrainer(cfg.train)
        fit_summary = trainer.fit()
    elif cfg.execution.run_live:
        if not cfg.live.artifact_bundle_path:
            raise ValueError(
                "run_train is False and run_live is True, so live.artifact_bundle_path must be provided."
            )

    # -------------------------------------------------
    # Step 3: live inference using only the artifact bundle
    # -------------------------------------------------
    if cfg.execution.run_live:
        live_cfg = deepcopy(cfg.live)

        if not live_cfg.artifact_bundle_path:
            if fit_summary is None:
                raise RuntimeError("No fit summary available to resolve artifact bundle path.")
            artifact_bundle_path = fit_summary.get("artifact_bundle_path")
            if not isinstance(artifact_bundle_path, str) or not artifact_bundle_path:
                raise RuntimeError("Training did not return a valid artifact_bundle_path.")
            live_cfg.artifact_bundle_path = artifact_bundle_path

        if cfg.execution.use_training_test_split_for_live:
            if fit_summary is None:
                raise RuntimeError(
                    "use_training_test_split_for_live=True requires run_train=True in the same orchestration call."
                )

            split_paths = fit_summary.get("split_paths", {})
            test_raw_csv = split_paths.get("test_raw_csv")
            if not isinstance(test_raw_csv, str) or not test_raw_csv:
                raise RuntimeError(
                    "Training did not export split_paths['test_raw_csv']. "
                    "Set train.runtime.export_split_csvs=True."
                )

            live_cfg.live_root_path = os.path.dirname(test_raw_csv)
            live_cfg.live_data_path = os.path.basename(test_raw_csv)
        else:
            if not live_cfg.live_root_path or not live_cfg.live_data_path:
                raise ValueError(
                    "When use_training_test_split_for_live=False, set live.live_root_path and live.live_data_path."
                )

        live_service = LSTDLiveInferenceService(live_cfg)

        live_mode = cfg.execution.live_mode.lower()
        if live_mode == "static":
            live_summary = live_service.run_static()
        elif live_mode == "online":
            live_summary = live_service.run_online()
        else:
            raise ValueError("execution.live_mode must be one of: online, static")

    # -------------------------------------------------
    # Final summary
    # -------------------------------------------------
    summary: Dict[str, Any] = {
        "download_result": download_result,
        "fit_summary": fit_summary,
        "live_summary": live_summary,
    }

    summary_dir = _resolve_summary_dir(cfg, fit_summary, live_summary)
    ensure_dir(summary_dir)
    summary_path = os.path.join(summary_dir, cfg.execution.pipeline_summary_name)
    save_json(summary_path, summary)
    summary["pipeline_summary_path"] = summary_path
    return summary


def _resolve_summary_dir(
    cfg: PipelineConfig,
    fit_summary: Optional[Dict[str, Any]],
    live_summary: Optional[Dict[str, Any]],
) -> str:
    if fit_summary is not None:
        run_id = fit_summary.get("run_id")
        if isinstance(run_id, str) and run_id:
            return os.path.join(cfg.train.runtime.outputs_dir, run_id)

    if live_summary is not None:
        run_id = live_summary.get("run_id")
        if isinstance(run_id, str) and run_id:
            return os.path.join(cfg.live.runtime.outputs_dir, run_id)

    return cfg.train.runtime.outputs_dir