# run_train.py

import os   

from lstd_aws.binanace_history.config import HistoricalDownloadConfig
from lstd_aws.binanace_history.pipeline import download_historical_klines
from lstd_aws.training.config import FitTrainConfig
from lstd_aws.training.trainer import LSTDFitTrainer

download_cfg = HistoricalDownloadConfig(
    symbol="BTCUSDT",
    interval="1m",
    lookback_days=240,   # ~8 months
    output_dir="data",
    validate_contiguity=True,
    allow_missing_candles=False,
)

download_result = download_historical_klines(download_cfg)

cfg = FitTrainConfig(
    root_path="data",
    data_path=os.path.basename(download_result["raw_path"]),
)

# all data -> training
cfg.windows.seq_len = 17
cfg.windows.label_len = 16
cfg.windows.pred_len = 1

cfg.windows.train_ratio = 1.0
cfg.windows.val_ratio = 0.0
cfg.windows.test_ratio = 0.0

cfg.windows.features = "MS"
cfg.windows.target = "close"
cfg.windows.scale = True
cfg.windows.inverse = False
cfg.windows.timeenc = 2
cfg.windows.freq = "1min"
cfg.windows.delay_fb = False

cfg.model.mode = "feature"
cfg.model.long_conv_hidden = 640
cfg.model.short_mlp_hidden = 512
cfg.model.future_mlp_hidden = 512
cfg.model.lags = 1
cfg.model.prior_hidden_dim = 128
cfg.model.prior_num_hidden_layers = 3
cfg.model.zc_kl_weight = 1.0
cfg.model.zd_kl_weight = 1.0
cfg.model.L1_weight = 0.0
cfg.model.L2_weight = 1e-2

cfg.optim.train_epochs = 1
cfg.optim.batch_size = 32
cfg.optim.val_batch_size = 1
cfg.optim.learning_rate = 1e-3
cfg.optim.weight_decay = 0.0
cfg.optim.grad_clip_norm = 1.0
cfg.optim.use_amp = False
cfg.optim.num_workers = 0
cfg.optim.pin_memory = False

cfg.runtime.device = "cuda"
cfg.runtime.experiment_name = "btc_lstd_full_train"
cfg.runtime.checkpoints_dir = "checkpoints"
cfg.runtime.outputs_dir = "outputs"
cfg.runtime.export_split_csvs = False

cfg.s3_artifacts.enabled = True
cfg.s3_artifacts.bucket = "my-lstd-data"
cfg.s3_artifacts.region = "eu-central-1"
cfg.s3_artifacts.prefix = "training-artifacts"

cfg.patience = 1

trainer = LSTDFitTrainer(cfg)
summary = trainer.fit()
print(summary)