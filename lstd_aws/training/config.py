# training/config.py

from dataclasses import dataclass, field
from typing import Optional

from datasets.lstd_dataset import FeatureMode
from feature_engineering.config import FeatureConfig, LSTDExportConfig
from lstd_core.config import LSTDModelConfig


@dataclass
class WindowConfig:
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 16

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    features: FeatureMode = "M"
    target: str = "close"
    scale: bool = True
    inverse: bool = False
    timeenc: int = 0
    freq: str = "1min"
    delay_fb: bool = False


@dataclass
class OptimizerConfig:
    train_epochs: int = 20
    batch_size: int = 32
    val_batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = None
    num_workers: int = 0
    pin_memory: bool = False
    use_amp: bool = False
    print_every: int = 50


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    checkpoints_dir: str = "checkpoints"
    outputs_dir: str = "outputs"
    experiment_name: str = "lstd_btc_fit"
    export_split_csvs: bool = True
    artifact_bundle_name: str = "artifact_bundle.pt"


@dataclass
class FitTrainConfig:
    # This is now the RAW historical csv, not a pre-engineered lstd csv.
    root_path: str = "data"
    data_path: str = ""

    windows: WindowConfig = field(default_factory=WindowConfig)
    model: LSTDModelConfig = field(default_factory=LSTDModelConfig)
    feature_pipeline: FeatureConfig = field(default_factory=FeatureConfig)
    export: LSTDExportConfig = field(default_factory=LSTDExportConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    patience: int = 5
