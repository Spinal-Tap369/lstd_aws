# live_inference/config.py

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvaluationConfig:
    compute_unscaled_metrics: bool = True


@dataclass
class AdaptationConfig:
    mode: str = "full"  # none | full | regressor
    n_inner: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None
    use_amp: bool = False
    require_batch_size_one: bool = True
    save_adapted_checkpoint: bool = True


@dataclass
class GapFillConfig:
    enabled: bool = False
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    base_url: str = "https://api.binance.com"
    timeout: int = 20
    max_retries: int = 5
    request_limit: int = 1000
    sleep_seconds: float = 0.15


@dataclass
class RuntimeConfig:
    device: str = "auto"
    outputs_dir: str = "live_outputs"
    run_name: str = "lstd_live_inference"
    save_arrays: bool = True
    save_state_snapshot: bool = True


@dataclass
class LiveInferenceConfig:
    artifact_bundle_path: str = ""
    resume_state_path: str = ""

    # Optional local/Kaggle raw live input source.
    live_root_path: str = ""
    live_data_path: str = ""

    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    adapt: AdaptationConfig = field(default_factory=AdaptationConfig)
    gap_fill: GapFillConfig = field(default_factory=GapFillConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
