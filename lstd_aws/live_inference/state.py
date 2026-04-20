# live_inference/state.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PendingPredictionState:
    batch_x: list[list[float]]
    label_context: list[list[float]]
    predicted_seq: list[list[float]]
    future_rows: list[list[float]] = field(default_factory=list)
    issued_after_open_time: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_x": self.batch_x,
            "label_context": self.label_context,
            "predicted_seq": self.predicted_seq,
            "future_rows": self.future_rows,
            "issued_after_open_time": self.issued_after_open_time,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PendingPredictionState":
        return cls(
            batch_x=list(payload.get("batch_x", [])),
            label_context=list(payload.get("label_context", [])),
            predicted_seq=list(payload.get("predicted_seq", [])),
            future_rows=list(payload.get("future_rows", [])),
            issued_after_open_time=(
                None if payload.get("issued_after_open_time") is None else int(payload["issued_after_open_time"])
            ),
        )


@dataclass
class LiveRuntimeState:
    history_scaled_rows: list[list[float]] = field(default_factory=list)
    history_dates: list[str] = field(default_factory=list)
    pending_predictions: list[PendingPredictionState] = field(default_factory=list)
    feature_pipeline_state: dict[str, Any] = field(default_factory=dict)
    last_completed_open_time: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_scaled_rows": self.history_scaled_rows,
            "history_dates": self.history_dates,
            "pending_predictions": [item.to_dict() for item in self.pending_predictions],
            "feature_pipeline_state": self.feature_pipeline_state,
            "last_completed_open_time": self.last_completed_open_time,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiveRuntimeState":
        return cls(
            history_scaled_rows=list(payload.get("history_scaled_rows", [])),
            history_dates=list(payload.get("history_dates", [])),
            pending_predictions=[
                PendingPredictionState.from_dict(item) for item in payload.get("pending_predictions", [])
            ],
            feature_pipeline_state=dict(payload.get("feature_pipeline_state", {})),
            last_completed_open_time=(
                None if payload.get("last_completed_open_time") is None else int(payload["last_completed_open_time"])
            ),
        )
