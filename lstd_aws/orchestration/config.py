# orchestration/config.py

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

from binanace_history.config import HistoricalDownloadConfig
from live_inference.config import LiveInferenceConfig
from training.config import FitTrainConfig


@dataclass
class ExecutionConfig:
    run_download: bool = True
    run_train: bool = True
    run_live: bool = True
    live_mode: str = "online"  # online | static
    use_training_test_split_for_live: bool = True
    pipeline_summary_name: str = "pipeline_summary.json"


@dataclass
class LocalReplayConfig:
    """
    Single-machine validation harness.

    Why this exists:
    - service mode tests artifact handoff directly through LSTDLiveInferenceService
    - worker mode tests the sqs-worker style behavior locally:
      catch up the gap, optionally warm backlog, then only start prediction/adaptation

    Important:
    - skip_bars lets you simulate inference starting late by several bars/days
    - initial_queue_backlog_bars lets you simulate the worker seeing a small backlog
      already waiting in the queue before it starts doing real live forecasting
    """

    enabled: bool = False
    mode: str = "worker"  # worker | service
    skip_bars: int = 0
    initial_queue_backlog_bars: int = 0
    max_rows: int = 0
    output_dir_name: str = "local_replay"
    derived_live_csv_name: str = "derived_live_input.csv"
    force_gap_fill_for_service: bool = True


@dataclass
class PipelineConfig:
    download: HistoricalDownloadConfig = field(default_factory=HistoricalDownloadConfig)
    train: FitTrainConfig = field(default_factory=FitTrainConfig)
    live: LiveInferenceConfig = field(default_factory=LiveInferenceConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    local_replay: LocalReplayConfig = field(default_factory=LocalReplayConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg = cls()
        _merge_dataclass(cfg, payload)
        return cfg

    @classmethod
    def default_btc_1m_real(cls) -> "PipelineConfig":
        cfg = cls()

        # -------------------------------------------------
        # Download
        # -------------------------------------------------
        cfg.download.symbol = "BTCUSDT"
        cfg.download.interval = "1m"
        cfg.download.lookback_days = 730
        cfg.download.output_dir = "data"
        cfg.download.validate_contiguity = True
        cfg.download.allow_missing_candles = False

        # -------------------------------------------------
        # Windows / split
        # -------------------------------------------------
        cfg.train.windows.seq_len = 17
        cfg.train.windows.label_len = 16
        cfg.train.windows.pred_len = 1
        cfg.train.windows.train_ratio = 0.25
        cfg.train.windows.val_ratio = 0.05
        cfg.train.windows.test_ratio = 0.70
        cfg.train.windows.features = "MS"
        cfg.train.windows.target = "close"
        cfg.train.windows.scale = True
        cfg.train.windows.inverse = False
        cfg.train.windows.timeenc = 2
        cfg.train.windows.freq = "1min"
        cfg.train.windows.delay_fb = False

        # -------------------------------------------------
        # Model
        # -------------------------------------------------
        cfg.train.model.mode = "feature"
        cfg.train.model.long_conv_hidden = 640
        cfg.train.model.short_mlp_hidden = 512
        cfg.train.model.future_mlp_hidden = 512
        cfg.train.model.lags = 1
        cfg.train.model.prior_hidden_dim = 128
        cfg.train.model.prior_num_hidden_layers = 3
        cfg.train.model.zc_kl_weight = 1.0
        cfg.train.model.zd_kl_weight = 1.0
        cfg.train.model.L1_weight = 0.0
        cfg.train.model.L2_weight = 1e-2

        # -------------------------------------------------
        # Optimizer / training
        # -------------------------------------------------
        cfg.train.optim.train_epochs = 1
        cfg.train.optim.batch_size = 32
        cfg.train.optim.val_batch_size = 1
        cfg.train.optim.learning_rate = 1e-3
        cfg.train.optim.weight_decay = 0.0
        cfg.train.optim.grad_clip_norm = 1.0
        cfg.train.optim.use_amp = False
        cfg.train.optim.num_workers = 0
        cfg.train.optim.pin_memory = False

        # -------------------------------------------------
        # Runtime
        # -------------------------------------------------
        cfg.train.runtime.device = "cuda"
        cfg.train.runtime.experiment_name = "btc_lstd"
        cfg.train.runtime.export_split_csvs = True
        cfg.train.runtime.checkpoints_dir = "checkpoints"
        cfg.train.runtime.outputs_dir = "outputs"
        cfg.train.runtime.artifact_bundle_name = "artifact_bundle.pt"

        cfg.train.patience = 1

        # -------------------------------------------------
        # Feature engineering defaults
        # -------------------------------------------------
        cfg.train.feature_pipeline.add_basic_price_features = True
        cfg.train.feature_pipeline.add_instance_norm_features = False
        cfg.train.feature_pipeline.add_seasonal_trend_features = True
        cfg.train.feature_pipeline.seasonal_trend_window = 96
        cfg.train.feature_pipeline.add_frequency_features = True
        cfg.train.feature_pipeline.frequency_window = 128
        cfg.train.feature_pipeline.add_long_short_regime_features = True
        cfg.train.feature_pipeline.short_window = 16
        cfg.train.feature_pipeline.long_window = 96
        cfg.train.feature_pipeline.add_explicit_targets = False
        cfg.train.feature_pipeline.drop_na_rows = True

        cfg.train.export.enabled = True
        cfg.train.export.feature_mode = "MS"
        cfg.train.export.target_column = "close"
        cfg.train.export.drop_explicit_target_columns = True

        # -------------------------------------------------
        # Live inference
        # -------------------------------------------------
        cfg.live.artifact_bundle_path = ""
        cfg.live.resume_state_path = ""
        cfg.live.live_root_path = ""
        cfg.live.live_data_path = ""

        cfg.live.eval.compute_unscaled_metrics = True

        cfg.live.adapt.mode = "full"
        cfg.live.adapt.n_inner = 1
        cfg.live.adapt.learning_rate = 1e-3
        cfg.live.adapt.weight_decay = 0.0
        cfg.live.adapt.grad_clip_norm = 1.0
        cfg.live.adapt.use_amp = False
        cfg.live.adapt.require_batch_size_one = True
        cfg.live.adapt.save_adapted_checkpoint = True

        cfg.live.gap_fill.enabled = False
        cfg.live.gap_fill.symbol = "BTCUSDT"
        cfg.live.gap_fill.interval = "1m"

        cfg.live.runtime.device = "cuda"
        cfg.live.runtime.outputs_dir = "live_outputs"
        cfg.live.runtime.run_name = "btc_lstd_live"
        cfg.live.runtime.save_arrays = True
        cfg.live.runtime.save_state_snapshot = True

        # -------------------------------------------------
        # Orchestration
        # -------------------------------------------------
        cfg.execution.run_download = True
        cfg.execution.run_train = True
        cfg.execution.run_live = True
        cfg.execution.live_mode = "online"
        cfg.execution.use_training_test_split_for_live = True
        cfg.execution.pipeline_summary_name = "pipeline_summary.json"

        # -------------------------------------------------
        # Local replay harness
        # -------------------------------------------------
        cfg.local_replay.enabled = False
        cfg.local_replay.mode = "worker"
        cfg.local_replay.skip_bars = 0
        cfg.local_replay.initial_queue_backlog_bars = 0
        cfg.local_replay.max_rows = 0
        cfg.local_replay.output_dir_name = "local_replay"
        cfg.local_replay.derived_live_csv_name = "derived_live_input.csv"
        cfg.local_replay.force_gap_fill_for_service = True

        return cfg


def _merge_dataclass(instance: Any, payload: dict[str, Any]) -> Any:
    for f in fields(instance):
        if f.name not in payload:
            continue

        current_value = getattr(instance, f.name)
        incoming_value = payload[f.name]

        if is_dataclass(current_value) and isinstance(incoming_value, dict):
            _merge_dataclass(current_value, incoming_value)
        else:
            setattr(instance, f.name, deepcopy(incoming_value))

    return instance
