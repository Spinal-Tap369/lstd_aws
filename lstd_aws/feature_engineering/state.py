# feature_engineering/state.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from .config import FeatureConfig
from .engineering import apply_feature_pipeline, compute_required_history_bars


@dataclass
class FeaturePipelineState:
    required_history_bars: int
    last_open_time: Optional[int]
    raw_history_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_history_bars": int(self.required_history_bars),
            "last_open_time": None if self.last_open_time is None else int(self.last_open_time),
            "raw_history_records": list(self.raw_history_records),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeaturePipelineState":
        return cls(
            required_history_bars=int(payload["required_history_bars"]),
            last_open_time=(None if payload.get("last_open_time") is None else int(payload["last_open_time"])),
            raw_history_records=list(payload.get("raw_history_records", [])),
        )


class StatefulFeatureEngineer:
    """
    Keeps enough raw history to recompute rolling features for new live bars.

    The preserved state is raw-tail based, not opaque-feature based. That makes
    the live side deterministic, debuggable, and easy to resume on AWS.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self.required_history_bars = int(compute_required_history_bars(cfg))

    def transform_full(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return raw_df.copy()

        clean = self._normalize_raw_df(raw_df)
        return apply_feature_pipeline(clean, self.cfg)

    def build_state_from_raw(self, raw_df: pd.DataFrame) -> FeaturePipelineState:
        if raw_df.empty:
            return FeaturePipelineState(
                required_history_bars=self.required_history_bars,
                last_open_time=None,
                raw_history_records=[],
            )

        clean = self._normalize_raw_df(raw_df)
        tail_df = clean.tail(self.required_history_bars).copy().reset_index(drop=True)
        records = self._raw_df_to_records(tail_df)
        last_open_time = int(clean["open_time"].iloc[-1])

        return FeaturePipelineState(
            required_history_bars=self.required_history_bars,
            last_open_time=last_open_time,
            raw_history_records=records,
        )

    def transform_incremental(
        self,
        new_raw_df: pd.DataFrame,
        state: FeaturePipelineState,
    ) -> tuple[pd.DataFrame, FeaturePipelineState]:
        if new_raw_df.empty:
            return pd.DataFrame(), state

        history_df = self._records_to_raw_df(state.raw_history_records)
        incoming_df = self._normalize_raw_df(new_raw_df)

        if state.last_open_time is not None:
            incoming_df = incoming_df[incoming_df["open_time"] > state.last_open_time].copy()

        if incoming_df.empty:
            return pd.DataFrame(), state

        combined = pd.concat([history_df, incoming_df], ignore_index=True)
        combined = self._normalize_raw_df(combined)

        feat_df = apply_feature_pipeline(combined, self.cfg)
        cutoff = state.last_open_time
        if cutoff is None:
            new_feat_df = feat_df.copy()
        else:
            new_feat_df = feat_df[feat_df["open_time"] > cutoff].copy()

        new_feat_df = new_feat_df.reset_index(drop=True)
        new_state = self.build_state_from_raw(combined)
        return new_feat_df, new_state

    def _normalize_raw_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return raw_df.copy()

        out = raw_df.copy()
        if "open_time" not in out.columns:
            raise ValueError("Raw dataframe must contain 'open_time'.")

        out = out.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

        if "open_dt" in out.columns:
            out["open_dt"] = pd.to_datetime(out["open_dt"], utc=True, errors="coerce")
        if "close_dt" in out.columns:
            out["close_dt"] = pd.to_datetime(out["close_dt"], utc=True, errors="coerce")

        return out

    def _raw_df_to_records(self, raw_df: pd.DataFrame) -> list[dict[str, Any]]:
        if raw_df.empty:
            return []

        out = raw_df.copy()
        for col in ["open_dt", "close_dt"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S%z")
        return out.to_dict(orient="records")

    def _records_to_raw_df(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        for col in ["open_dt", "close_dt"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        return self._normalize_raw_df(df)
