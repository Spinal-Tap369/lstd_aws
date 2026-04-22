# orchestration/run.py

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional

import pandas as pd

from binanace_history.pipeline import download_historical_klines
from live_inference.service import LSTDLiveInferenceService
from training.trainer import LSTDFitTrainer
from training.utils import ensure_dir, save_json

from .config import PipelineConfig
from .local_worker_replay import LocalWorkerReplay


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
    local_replay_info: Optional[Dict[str, Any]] = None

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

        base_live_csv = _resolve_base_live_csv(cfg, fit_summary, live_cfg)

        if cfg.local_replay.enabled:
            local_replay_info = _materialize_local_replay_csv(
                cfg=cfg,
                fit_summary=fit_summary,
                base_live_csv=base_live_csv,
            )
            derived_live_csv = local_replay_info["derived_live_csv_path"]

            if cfg.local_replay.mode.lower() == "service":
                live_cfg.live_root_path = os.path.dirname(derived_live_csv)
                live_cfg.live_data_path = os.path.basename(derived_live_csv)

                if cfg.local_replay.force_gap_fill_for_service:
                    live_cfg.gap_fill.enabled = True
                    live_cfg.gap_fill.symbol = cfg.download.symbol
                    live_cfg.gap_fill.interval = cfg.download.interval

                live_service = LSTDLiveInferenceService(live_cfg)
                live_mode = cfg.execution.live_mode.lower()
                if live_mode == "static":
                    live_summary = live_service.run_static()
                elif live_mode == "online":
                    live_summary = live_service.run_online()
                else:
                    raise ValueError("execution.live_mode must be one of: online, static")

            elif cfg.local_replay.mode.lower() == "worker":
                replay = LocalWorkerReplay(
                    live_cfg,
                    raw_csv_path=derived_live_csv,
                    symbol=cfg.download.symbol,
                    interval=cfg.download.interval,
                    initial_queue_backlog_bars=cfg.local_replay.initial_queue_backlog_bars,
                )
                live_summary = replay.run()
            else:
                raise ValueError("local_replay.mode must be one of: worker, service")

        else:
            live_cfg.live_root_path = os.path.dirname(base_live_csv)
            live_cfg.live_data_path = os.path.basename(base_live_csv)

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
        "local_replay_info": local_replay_info,
    }

    summary_dir = _resolve_summary_dir(cfg, fit_summary, live_summary)
    ensure_dir(summary_dir)
    summary_path = os.path.join(summary_dir, cfg.execution.pipeline_summary_name)
    save_json(summary_path, summary)
    summary["pipeline_summary_path"] = summary_path
    return summary


def _resolve_base_live_csv(
    cfg: PipelineConfig,
    fit_summary: Optional[Dict[str, Any]],
    live_cfg,
) -> str:
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
        return test_raw_csv

    if not live_cfg.live_root_path or not live_cfg.live_data_path:
        raise ValueError(
            "When use_training_test_split_for_live=False, set live.live_root_path and live.live_data_path."
        )

    return os.path.join(live_cfg.live_root_path, live_cfg.live_data_path)


def _materialize_local_replay_csv(
    *,
    cfg: PipelineConfig,
    fit_summary: Optional[Dict[str, Any]],
    base_live_csv: str,
) -> Dict[str, Any]:
    raw_df = pd.read_csv(base_live_csv)
    if raw_df.empty:
        raise RuntimeError("Base live CSV is empty")

    skip_bars = int(cfg.local_replay.skip_bars)
    if skip_bars < 0:
        raise ValueError("local_replay.skip_bars must be >= 0")
    if skip_bars >= len(raw_df):
        raise ValueError(
            f"local_replay.skip_bars={skip_bars} is too large for live source with {len(raw_df)} rows"
        )

    derived_df = raw_df.iloc[skip_bars:].copy().reset_index(drop=True)

    max_rows = int(cfg.local_replay.max_rows)
    if max_rows > 0:
        derived_df = derived_df.head(max_rows).copy().reset_index(drop=True)

    if derived_df.empty:
        raise RuntimeError("Derived local replay CSV is empty after skip/max_rows")

    summary_dir = _resolve_summary_dir(cfg, fit_summary, None)
    replay_dir = os.path.join(summary_dir, cfg.local_replay.output_dir_name)
    ensure_dir(replay_dir)

    derived_csv_path = os.path.join(replay_dir, cfg.local_replay.derived_live_csv_name)
    derived_df.to_csv(derived_csv_path, index=False)

    first_open_time = int(derived_df.iloc[0]["open_time"])
    last_open_time = int(derived_df.iloc[-1]["open_time"])

    info = {
        "mode": cfg.local_replay.mode,
        "base_live_csv_path": base_live_csv,
        "derived_live_csv_path": derived_csv_path,
        "skip_bars": skip_bars,
        "initial_queue_backlog_bars": int(cfg.local_replay.initial_queue_backlog_bars),
        "max_rows": max_rows,
        "rows_written": int(len(derived_df)),
        "first_open_time": first_open_time,
        "last_open_time": last_open_time,
    }

    save_json(os.path.join(replay_dir, "local_replay_info.json"), info)
    return info


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
