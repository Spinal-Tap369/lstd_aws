# datasets/time_features.py

import numpy as np
import pandas as pd


def _minute_divisor_from_freq(freq: str) -> int:
    f = str(freq).strip().lower()
    if f in {"1min", "1m", "t", "1t"}:
        return 1
    if f in {"3min", "3m", "3t"}:
        return 3
    if f in {"5min", "5m", "5t"}:
        return 5
    if f in {"15min", "15m", "15t"}:
        return 15
    if f in {"30min", "30m", "30t"}:
        return 30
    if f in {"1h", "h", "2h", "4h", "6h", "8h", "12h", "1d", "d"}:
        return 60
    return 60


def time_features(df_stamp: pd.DataFrame, timeenc: int = 0, freq: str = "1min") -> np.ndarray:
    if "date" not in df_stamp.columns:
        raise ValueError("df_stamp must contain a 'date' column")

    dt = pd.to_datetime(df_stamp["date"])

    month = np.asarray(dt.dt.month.to_numpy(), dtype=np.int64)
    day = np.asarray(dt.dt.day.to_numpy(), dtype=np.int64)
    weekday = np.asarray(dt.dt.weekday.to_numpy(), dtype=np.int64)
    hour = np.asarray(dt.dt.hour.to_numpy(), dtype=np.int64)
    minute = np.asarray(dt.dt.minute.to_numpy(), dtype=np.int64)

    if timeenc == 0:
        divisor = _minute_divisor_from_freq(freq)
        minute_bucket = (minute // divisor).astype(np.int64)
        return np.stack([month, day, weekday, hour, minute_bucket], axis=1).astype(np.float32)

    return np.stack(
        [
            (month - 1).astype(np.float32) / 11.0,
            (day - 1).astype(np.float32) / 30.0,
            weekday.astype(np.float32) / 6.0,
            hour.astype(np.float32) / 23.0,
            minute.astype(np.float32) / 59.0,
        ],
        axis=1,
    ).astype(np.float32)