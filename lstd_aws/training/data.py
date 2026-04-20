# training/data.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from datasets.scalers import StandardScalerNumpy
from datasets.time_features import time_features
from feature_engineering.export import build_lstd_ready_frame
from feature_engineering.state import StatefulFeatureEngineer
from .config import FitTrainConfig
from .utils import ensure_dir


class EngineeredWindowDataset(Dataset):
    def __init__(
        self,
        lstd_df: pd.DataFrame,
        model_columns: list[str],
        scaler: StandardScalerNumpy,
        *,
        seq_len: int,
        label_len: int,
        pred_len: int,
        features: str,
        target: str,
        scale: bool,
        inverse: bool,
        timeenc: int,
        freq: str,
        effective_start_idx: int = 0,
        effective_end_idx: Optional[int] = None,
        date_scaler: Optional[StandardScalerNumpy] = None,
    ) -> None:
        if "date" not in lstd_df.columns:
            raise ValueError("Expected 'date' column in engineered LSTD dataframe.")
        if target not in model_columns:
            raise ValueError(f"Target '{target}' must be present in model_columns.")

        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.features = features
        self.target = target
        self.scale = bool(scale)
        self.inverse = bool(inverse)
        self.timeenc = int(timeenc)
        self.freq = freq
        self.model_columns = list(model_columns)
        self.scaler = scaler
        self.date_scaler = date_scaler

        self.data_x = np.empty((0, 0), dtype=np.float32)
        self.data_y = np.empty((0, 0), dtype=np.float32)
        self.data_stamp = np.empty((0, 0), dtype=np.float32)
        self.valid_indices: list[int] = []

        df = lstd_df[["date"] + self.model_columns].copy().reset_index(drop=True)
        data_values = df[self.model_columns].to_numpy(dtype=np.float32, copy=True)
        data = scaler.transform(data_values) if self.scale else data_values

        stamp_df = pd.DataFrame({"date": pd.to_datetime(df["date"], errors="coerce")})
        data_stamp = time_features(stamp_df, timeenc=self.timeenc, freq=self.freq)
        if self.timeenc == 2 and self.date_scaler is not None:
            data_stamp = self.date_scaler.transform(data_stamp)

        self.data_x = np.asarray(data, dtype=np.float32)
        self.data_y = np.asarray(data_values if self.inverse else data, dtype=np.float32)
        self.data_stamp = np.asarray(data_stamp, dtype=np.float32)

        end_idx = len(df) if effective_end_idx is None else int(effective_end_idx)
        start_idx = int(effective_start_idx)
        max_start = len(df) - self.seq_len - self.pred_len + 1
        for s_begin in range(max(0, max_start)):
            s_end = s_begin + self.seq_len
            target_start = s_end
            target_end = s_end + self.pred_len
            if target_start >= start_idx and target_end <= end_idx:
                self.valid_indices.append(s_begin)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int):
        s_begin = self.valid_indices[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]],
                axis=0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark


@dataclass
class TrainValLoaderBundle:
    train_dataset: EngineeredWindowDataset
    train_loader: DataLoader
    val_dataset: Optional[EngineeredWindowDataset]
    val_loader: Optional[DataLoader]
    enc_in: int
    target_index: int
    model_columns: list[str]
    target_column: str
    input_scaler: StandardScalerNumpy
    date_scaler: Optional[StandardScalerNumpy]
    split_paths: dict[str, str]
    split_meta: dict[str, Any]
    initial_live_state: dict[str, Any]


@dataclass
class RawSplitFrames:
    train_raw: pd.DataFrame
    val_raw: pd.DataFrame
    test_raw: pd.DataFrame
    raw_df: pd.DataFrame
    n_train: int
    n_val: int
    n_test: int


def export_chronological_splits(cfg: FitTrainConfig) -> dict[str, Any]:
    full_path = os.path.join(cfg.root_path, cfg.data_path)
    raw_df = pd.read_csv(full_path)

    tr = cfg.windows.train_ratio
    vr = cfg.windows.val_ratio
    te = cfg.windows.test_ratio
    if abs((tr + vr + te) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n_rows = len(raw_df)
    n_train = int(n_rows * tr)
    n_val = int(n_rows * vr)

    train_raw = raw_df.iloc[:n_train].copy()
    val_raw = raw_df.iloc[n_train:n_train + n_val].copy()
    test_raw = raw_df.iloc[n_train + n_val:].copy()

    out_dir = os.path.join(cfg.runtime.outputs_dir, cfg.runtime.experiment_name, "raw_splits")
    ensure_dir(out_dir)

    stem = os.path.splitext(os.path.basename(cfg.data_path))[0]
    paths = {
        "train_raw_csv": os.path.join(out_dir, f"{stem}_train_raw.csv"),
        "val_raw_csv": os.path.join(out_dir, f"{stem}_val_raw.csv"),
        "test_raw_csv": os.path.join(out_dir, f"{stem}_test_raw.csv"),
    }

    train_raw.to_csv(paths["train_raw_csv"], index=False)
    val_raw.to_csv(paths["val_raw_csv"], index=False)
    test_raw.to_csv(paths["test_raw_csv"], index=False)

    return {
        "paths": paths,
        "meta": {
            "rows_total": n_rows,
            "rows_train": len(train_raw),
            "rows_val": len(val_raw),
            "rows_test": len(test_raw),
            "train_ratio": tr,
            "val_ratio": vr,
            "test_ratio": te,
        },
    }


def build_train_val_loaders(cfg: FitTrainConfig) -> TrainValLoaderBundle:
    split_export = export_chronological_splits(cfg) if cfg.runtime.export_split_csvs else _build_in_memory_split_export(cfg)
    split_frames = _load_raw_split_frames(cfg)

    feature_engineer = StatefulFeatureEngineer(cfg.feature_pipeline)
    required_history_bars = max(feature_engineer.required_history_bars, cfg.windows.seq_len)

    train_feat_df = feature_engineer.transform_full(split_frames.train_raw)
    if train_feat_df.empty:
        raise RuntimeError("Training feature engineering produced no rows. Increase training data or reduce feature lookback.")

    val_dataset = None
    val_loader = None

    export_cfg = _resolved_export_config(cfg)
    train_lstd_df, model_columns = build_lstd_ready_frame(train_feat_df, export_cfg)
    target_column = cfg.windows.target

    input_scaler = StandardScalerNumpy().fit(train_lstd_df[model_columns].to_numpy(dtype=np.float32, copy=True))
    date_scaler = _fit_date_scaler(train_lstd_df, cfg.windows.timeenc, cfg.windows.freq)

    train_dataset = EngineeredWindowDataset(
        train_lstd_df,
        model_columns,
        input_scaler,
        seq_len=cfg.windows.seq_len,
        label_len=cfg.windows.label_len,
        pred_len=cfg.windows.pred_len,
        features=cfg.windows.features,
        target=target_column,
        scale=cfg.windows.scale,
        inverse=cfg.windows.inverse,
        timeenc=cfg.windows.timeenc,
        freq=cfg.windows.freq,
        effective_start_idx=0,
        effective_end_idx=len(train_lstd_df),
        date_scaler=date_scaler,
    )
    if len(train_dataset) <= 0:
        raise RuntimeError("Train dataset length is 0. Increase data range or reduce seq_len/pred_len.")

    if split_frames.n_val > 0:
        val_source_start = max(0, split_frames.n_train - required_history_bars)
        val_source_raw = split_frames.raw_df.iloc[val_source_start:split_frames.n_train + split_frames.n_val].copy()
        val_feat_df = feature_engineer.transform_full(val_source_raw)
        val_lstd_df, _ = build_lstd_ready_frame(val_feat_df, export_cfg)

        val_start_open_time = int(split_frames.val_raw.iloc[0]["open_time"])
        val_open_times = val_feat_df["open_time"].to_numpy(dtype=np.int64, copy=False)
        val_effective_start_idx = int(np.searchsorted(val_open_times, val_start_open_time, side="left"))

        val_dataset = EngineeredWindowDataset(
            val_lstd_df,
            model_columns,
            input_scaler,
            seq_len=cfg.windows.seq_len,
            label_len=cfg.windows.label_len,
            pred_len=cfg.windows.pred_len,
            features=cfg.windows.features,
            target=target_column,
            scale=cfg.windows.scale,
            inverse=cfg.windows.inverse,
            timeenc=cfg.windows.timeenc,
            freq=cfg.windows.freq,
            effective_start_idx=val_effective_start_idx,
            effective_end_idx=len(val_lstd_df),
            date_scaler=date_scaler,
        )
        if len(val_dataset) <= 0:
            raise RuntimeError("Val dataset length is 0. Increase data range or reduce seq_len/pred_len.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.optim.num_workers,
        drop_last=True,
        pin_memory=bool(cfg.optim.pin_memory),
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.optim.val_batch_size,
            shuffle=False,
            num_workers=cfg.optim.num_workers,
            drop_last=False,
            pin_memory=bool(cfg.optim.pin_memory),
        )

    warmup_raw = split_frames.raw_df.iloc[:split_frames.n_train + split_frames.n_val].copy()
    warmup_feat_df = feature_engineer.transform_full(warmup_raw)
    warmup_lstd_df, _ = build_lstd_ready_frame(warmup_feat_df, export_cfg)
    warmup_scaled = input_scaler.transform(warmup_lstd_df[model_columns].to_numpy(dtype=np.float32, copy=True))

    if len(warmup_scaled) < cfg.windows.seq_len:
        raise RuntimeError("Warm-up history is shorter than seq_len after feature engineering.")

    warmup_state = feature_engineer.build_state_from_raw(warmup_raw)
    history_scaled_rows = warmup_scaled[-cfg.windows.seq_len:].tolist()
    history_dates = warmup_lstd_df["date"].iloc[-cfg.windows.seq_len:].astype(str).tolist()
    last_completed_open_time = int(warmup_feat_df["open_time"].iloc[-1]) if not warmup_feat_df.empty else None

    initial_live_state = {
        "history_scaled_rows": history_scaled_rows,
        "history_dates": history_dates,
        "pending_predictions": [],
        "feature_pipeline_state": warmup_state.to_dict(),
        "last_completed_open_time": last_completed_open_time,
    }

    enc_in = len(model_columns)
    target_index = enc_in - 1 if enc_in > 0 else 0

    return TrainValLoaderBundle(
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_dataset=val_dataset,
        val_loader=val_loader,
        enc_in=enc_in,
        target_index=target_index,
        model_columns=model_columns,
        target_column=target_column,
        input_scaler=input_scaler,
        date_scaler=date_scaler,
        split_paths=split_export["paths"],
        split_meta=split_export["meta"],
        initial_live_state=initial_live_state,
    )


def _load_raw_split_frames(cfg: FitTrainConfig) -> RawSplitFrames:
    full_path = os.path.join(cfg.root_path, cfg.data_path)
    raw_df = pd.read_csv(full_path)
    if "open_time" not in raw_df.columns:
        raise ValueError("Raw historical csv must contain 'open_time'.")

    raw_df = raw_df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    tr = cfg.windows.train_ratio
    vr = cfg.windows.val_ratio
    te = cfg.windows.test_ratio
    if abs((tr + vr + te) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n_rows = len(raw_df)
    n_train = int(n_rows * tr)
    n_val = int(n_rows * vr)
    n_test = n_rows - n_train - n_val

    return RawSplitFrames(
        train_raw=raw_df.iloc[:n_train].copy(),
        val_raw=raw_df.iloc[n_train:n_train + n_val].copy(),
        test_raw=raw_df.iloc[n_train + n_val:].copy(),
        raw_df=raw_df,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )


def _fit_date_scaler(
    train_lstd_df: pd.DataFrame,
    timeenc: int,
    freq: str,
) -> Optional[StandardScalerNumpy]:
    if int(timeenc) != 2:
        return None

    stamp_df = pd.DataFrame({"date": pd.to_datetime(train_lstd_df["date"], errors="coerce")})
    stamp_values = time_features(stamp_df, timeenc=timeenc, freq=freq)
    scaler = StandardScalerNumpy().fit(stamp_values.astype(np.float32))
    return scaler


def _resolved_export_config(cfg: FitTrainConfig):
    export_cfg = cfg.export
    export_cfg.feature_mode = cfg.windows.features
    export_cfg.target_column = cfg.windows.target
    return export_cfg


def _build_in_memory_split_export(cfg: FitTrainConfig) -> dict[str, Any]:
    split_frames = _load_raw_split_frames(cfg)
    return {
        "paths": {},
        "meta": {
            "rows_total": len(split_frames.raw_df),
            "rows_train": len(split_frames.train_raw),
            "rows_val": len(split_frames.val_raw),
            "rows_test": len(split_frames.test_raw),
            "train_ratio": cfg.windows.train_ratio,
            "val_ratio": cfg.windows.val_ratio,
            "test_ratio": cfg.windows.test_ratio,
        },
    }
