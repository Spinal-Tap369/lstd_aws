# live_inference/telemetry.py

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import boto3


@dataclass
class TelemetryConfig:
    enabled: bool = False
    bucket: str = ""
    prefix: str = "live-telemetry"
    region: str = "eu-central-1"
    local_dir: str = "telemetry"


class S3TelemetryWriter:
    """
    Small helper for live-inference telemetry.

    Design choice:
    - every important worker event can be written locally
    - if enabled, the same payload is also uploaded to S3 as a small JSON object

    This is intentionally simple and durable. At 1m cadence, one-object-per-event is fine.
    """

    def __init__(
        self,
        cfg: TelemetryConfig,
        *,
        run_id: str,
        symbol: str,
        interval: str,
    ) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.symbol = symbol
        self.interval = interval

        os.makedirs(self.cfg.local_dir, exist_ok=True)
        self.local_events_path = os.path.join(self.cfg.local_dir, f"{self.run_id}_events.jsonl")
        self.local_latest_summary_path = os.path.join(self.cfg.local_dir, f"{self.run_id}_latest_summary.json")

        self.s3 = None
        if self.cfg.enabled:
            if not self.cfg.bucket:
                raise ValueError("TelemetryConfig.bucket is required when telemetry is enabled")
            self.s3 = boto3.client("s3", region_name=self.cfg.region)

    @classmethod
    def disabled(cls, *, run_id: str, symbol: str, interval: str) -> "S3TelemetryWriter":
        return cls(
            TelemetryConfig(enabled=False),
            run_id=run_id,
            symbol=symbol,
            interval=interval,
        )

    def _base_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "logged_at_epoch": int(time.time()),
        }

    def _event_s3_key(self, event_type: str, open_time: Optional[int]) -> str:
        ts_ms = int(time.time() * 1000)
        ot = "na" if open_time is None else str(int(open_time))
        prefix = self.cfg.prefix.strip("/")
        return (
            f"{prefix}/run_id={self.run_id}/symbol={self.symbol}/interval={self.interval}/"
            f"events/{ts_ms}_{event_type}_{ot}.json"
        )

    def _summary_s3_key(self) -> str:
        prefix = self.cfg.prefix.strip("/")
        return (
            f"{prefix}/run_id={self.run_id}/symbol={self.symbol}/interval={self.interval}/"
            "latest_summary.json"
        )

    def _write_local_json(self, path: str, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=self._json_default)

    def _append_local_jsonl(self, path: str, payload: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=self._json_default))
            handle.write("\n")

    def _put_s3_json(self, key: str, payload: dict[str, Any]) -> None:
        if self.s3 is None:
            return
        self.s3.put_object(
            Bucket=self.cfg.bucket,
            Key=key,
            Body=json.dumps(payload, default=self._json_default).encode("utf-8"),
            ContentType="application/json",
        )

    def log_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        open_time: Optional[int] = None,
    ) -> None:
        out = self._base_payload()
        out["event_type"] = event_type
        if open_time is not None:
            out["open_time"] = int(open_time)
        out.update(payload)

        self._append_local_jsonl(self.local_events_path, out)
        if self.cfg.enabled:
            self._put_s3_json(self._event_s3_key(event_type, open_time), out)

    def write_latest_summary(self, payload: dict[str, Any]) -> None:
        out = self._base_payload()
        out.update(payload)

        self._write_local_json(self.local_latest_summary_path, out)
        if self.cfg.enabled:
            self._put_s3_json(self._summary_s3_key(), out)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        try:
            import numpy as np
            import torch

            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if torch.is_tensor(obj):
                return obj.detach().cpu().tolist()
        except Exception:
            pass

        return str(obj)
