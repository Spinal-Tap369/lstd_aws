# datasets/lstd_dataset.py

from __future__ import annotations

import os
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .scalers import StandardScalerNumpy
from .time_features import time_features


SplitFlag = Literal["train", "val", "test"]
FeatureMode = Literal["S", "M", "MS"]


class LSTDDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: SplitFlag = "train",
        delay_fb: bool = False,
        size: Optional[Sequence[int]] = None,
        features: FeatureMode = "M",
        target: str = "close",
        scale: bool = True,
        inverse: bool = False,
        timeenc: int = 0,
        freq: str = "1min",
        cols: Optional[list[str]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 16
        else:
            if len(size) != 3:
                raise ValueError("size must contain exactly [seq_len, label_len, pred_len]")
            self.seq_len, self.label_len, self.pred_len = [int(v) for v in size]

        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be one of: train, val, test")
        if features not in {"S", "M", "MS"}:
            raise ValueError("features must be one of: S, M, MS")
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = int(timeenc)
        self.freq = freq
        self.delay_fb = bool(delay_fb)
        self.cols = list(cols) if cols is not None else None
        self.root_path = root_path
        self.data_path = data_path
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)

        self.scaler = StandardScalerNumpy()
        self.date_scaler = StandardScalerNumpy()

        self.data_x = np.empty((0, 0), dtype=np.float32)
        self.data_y = np.empty((0, 0), dtype=np.float32)
        self.data_stamp = np.empty((0, 0), dtype=np.float32)

        self.__read_data__()

    def __read_data__(self) -> None:
        full_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(full_path)

        if "date" not in df_raw.columns:
            raise ValueError("Dataset CSV must contain 'date' column")
        if self.target not in df_raw.columns:
            raise ValueError(f"Target '{self.target}' not found in CSV")

        if self.cols is not None:
            feature_cols = self.cols.copy()
            if self.target in feature_cols:
                feature_cols.remove(self.target)
        else:
            feature_cols = list(df_raw.columns)
            feature_cols.remove("date")
            feature_cols.remove(self.target)

        df_raw = df_raw[["date"] + feature_cols + [self.target]]

        n_rows = len(df_raw)
        n_train = int(n_rows * self.train_ratio)
        n_val = int(n_rows * self.val_ratio)

        border1s = [0, n_train - self.seq_len, n_train + n_val - self.seq_len]
        border2s = [n_train, n_train + n_val, n_rows]

        border1 = int(border1s[self.set_type])
        border2 = int(border2s[self.set_type])

        if self.features in {"M", "MS"}:
            df_data = df_raw.iloc[:, 1:]
        else:
            df_data = df_raw[[self.target]]

        data_values = df_data.to_numpy(dtype=np.float32, copy=True)

        if self.scale:
            train_data = data_values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data_values)
        else:
            data = data_values

        train_stamp = pd.DataFrame(
            {"date": pd.to_datetime(df_raw["date"].iloc[border1s[0]:border2s[0]], errors="coerce")}
        )
        train_mark = time_features(train_stamp, timeenc=self.timeenc, freq=self.freq)
        if self.timeenc == 2:
            self.date_scaler.fit(train_mark)

        df_stamp = pd.DataFrame(
            {"date": pd.to_datetime(df_raw["date"].iloc[border1:border2], errors="coerce")}
        )
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        if self.timeenc == 2:
            data_stamp = self.date_scaler.transform(data_stamp)

        self.data_x = np.asarray(data[border1:border2], dtype=np.float32)
        if self.inverse:
            self.data_y = np.asarray(data_values[border1:border2], dtype=np.float32)
        else:
            self.data_y = np.asarray(data[border1:border2], dtype=np.float32)
        self.data_stamp = np.asarray(data_stamp, dtype=np.float32)

    def __getitem__(self, index: int):
        if self.delay_fb and self.set_type == 2:
            s_begin = index * self.pred_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
        else:
            s_begin = index
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

    def __len__(self) -> int:
        if self.delay_fb and self.set_type == 2:
            return (len(self.data_x) - self.seq_len - self.pred_len) // self.pred_len
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)
