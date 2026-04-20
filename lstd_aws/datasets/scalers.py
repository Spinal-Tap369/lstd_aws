# datasets/scalers.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StandardScalerState:
    mean: list[list[float]]
    std: list[list[float]]


class StandardScalerNumpy:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScalerNumpy":
        mean = np.asarray(np.mean(x, axis=0, keepdims=True), dtype=np.float32)
        std = np.asarray(np.std(x, axis=0, keepdims=True), dtype=np.float32)
        std[std < 1e-12] = 1.0
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted")
        return x * self.std_ + self.mean_

    def to_state(self) -> StandardScalerState:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted")
        return StandardScalerState(
            mean=self.mean_.tolist(),
            std=self.std_.tolist(),
        )

    @classmethod
    def from_state(cls, state: StandardScalerState | dict) -> "StandardScalerNumpy":
        obj = cls()
        if isinstance(state, dict):
            mean = state["mean"]
            std = state["std"]
        else:
            mean = state.mean
            std = state.std
        obj.mean_ = np.asarray(mean, dtype=np.float32)
        obj.std_ = np.asarray(std, dtype=np.float32)
        return obj