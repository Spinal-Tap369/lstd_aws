# live_inference/sqs_worker.py

from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Optional, cast

import boto3
import numpy as np
import pandas as pd

from lstd_aws.binanace_history.client import BinanceHistoricalKlinesClient
from lstd_aws.training.metrics import regression_metrics

from .config import LiveInferenceConfig
from .service import LSTDLiveInferenceService
from .telemetry import S3TelemetryWriter, TelemetryConfig


LOGGER = logging.getLogger("lstd_sqs_worker")
STOP_REQUESTED = False
STEP_MS = 60_000


@dataclass
class WorkerConfig:
    aws_region: str
    raw_queue_url: str
    artifact_bundle_path: str
    resume_state_path: str
    live_symbol: str
    live_interval: str
    device: str
    outputs_dir: str
    run_name: str
    live_mode: str
    n_inner: int
    adapt_learning_rate: float
    adapt_weight_decay: float
    adapt_grad_clip_norm: Optional[float]
    compute_unscaled_metrics: bool
    save_arrays: bool
    save_state_snapshot: bool
    state_snapshot_name: str
    poll_wait_seconds: int
    backlog_drain_wait_seconds: int
    visibility_timeout_seconds: int
    stream_state_table: str
    gap_fill_base_url: str
    gap_fill_timeout: int
    gap_fill_max_retries: int
    gap_fill_request_limit: int
    gap_fill_sleep_seconds: float
    telemetry_enabled: bool
    telemetry_bucket: str
    telemetry_prefix: str
    telemetry_local_dir: str

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        raw_queue_url = _env_required("RAW_QUEUE_URL")
        artifact_bundle_path = _env_required("ARTIFACT_BUNDLE_PATH")

        live_symbol = os.environ.get("LIVE_SYMBOL", "BTCUSDT").strip().upper()
        live_interval = os.environ.get("LIVE_INTERVAL", "1m").strip()
        if live_interval != "1m":
            raise ValueError("This worker currently supports LIVE_INTERVAL=1m only")

        grad_clip_raw = os.environ.get("ADAPT_GRAD_CLIP_NORM", "1.0").strip()
        grad_clip_norm: Optional[float]
        if grad_clip_raw == "" or grad_clip_raw.lower() == "none":
            grad_clip_norm = None
        else:
            grad_clip_norm = float(grad_clip_raw)

        return cls(
            aws_region=os.environ.get("AWS_REGION", "eu-central-1").strip(),
            raw_queue_url=raw_queue_url,
            artifact_bundle_path=artifact_bundle_path,
            resume_state_path=os.environ.get("RESUME_STATE_PATH", "").strip(),
            live_symbol=live_symbol,
            live_interval=live_interval,
            device=os.environ.get("DEVICE", "auto").strip(),
            outputs_dir=os.environ.get("OUTPUTS_DIR", "live_outputs").strip(),
            run_name=os.environ.get("RUN_NAME", "lstd_live_worker").strip(),
            live_mode=os.environ.get("LIVE_MODE", "full").strip().lower(),
            n_inner=int(os.environ.get("N_INNER", "1")),
            adapt_learning_rate=float(os.environ.get("ADAPT_LEARNING_RATE", "1e-3")),
            adapt_weight_decay=float(os.environ.get("ADAPT_WEIGHT_DECAY", "0.0")),
            adapt_grad_clip_norm=grad_clip_norm,
            compute_unscaled_metrics=_env_bool("COMPUTE_UNSCALED_METRICS", True),
            save_arrays=_env_bool("SAVE_ARRAYS", False),
            save_state_snapshot=_env_bool("SAVE_STATE_SNAPSHOT", True),
            state_snapshot_name=os.environ.get("STATE_SNAPSHOT_NAME", "worker_latest_state.pt").strip(),
            poll_wait_seconds=int(os.environ.get("POLL_WAIT_SECONDS", "20")),
            backlog_drain_wait_seconds=int(os.environ.get("BACKLOG_DRAIN_WAIT_SECONDS", "1")),
            visibility_timeout_seconds=int(os.environ.get("VISIBILITY_TIMEOUT_SECONDS", "900")),
            stream_state_table=os.environ.get("STREAM_STATE_TABLE", "").strip(),
            gap_fill_base_url=os.environ.get("BINANCE_REST_BASE", "https://api.binance.com").strip(),
            gap_fill_timeout=int(os.environ.get("GAP_FILL_TIMEOUT", "20")),
            gap_fill_max_retries=int(os.environ.get("GAP_FILL_MAX_RETRIES", "5")),
            gap_fill_request_limit=int(os.environ.get("GAP_FILL_REQUEST_LIMIT", "1000")),
            gap_fill_sleep_seconds=float(os.environ.get("GAP_FILL_SLEEP_SECONDS", "0.15")),
            telemetry_enabled=_env_bool("TELEMETRY_ENABLED", False),
            telemetry_bucket=os.environ.get("TELEMETRY_BUCKET", "").strip(),
            telemetry_prefix=os.environ.get("TELEMETRY_PREFIX", "live-telemetry").strip(),
            telemetry_local_dir=os.environ.get("TELEMETRY_LOCAL_DIR", "telemetry").strip(),
        )


class LSTDSQSWorker:
    """
    FIFO SQS worker for live inference.

    Core behavior:
    1. Start from a training artifact bundle (and optional local resume snapshot).
    2. Consume raw candle messages from RAW_QUEUE_URL.
    3. If there is any gap from the artifact state to the first queued candle, fetch it from Binance REST.
    4. While catching up, do NOT predict and do NOT adapt.
    5. Drain any immediate queue backlog in the same warm-only way.
    6. Only when the queue is caught up and empty, arm the next forecast.
    7. From then on, do real live predict -> realize -> online-update.

    This matches your intended production behavior much better than trying to
    score stale historical backlog.
    """

    def __init__(self, cfg: WorkerConfig):
        self.cfg = cfg

        self.sqs = boto3.client("sqs", region_name=self.cfg.aws_region)
        self.ddb_table = None
        if self.cfg.stream_state_table:
            ddb = cast(Any, boto3.resource("dynamodb", region_name=self.cfg.aws_region))
            self.ddb_table = ddb.Table(self.cfg.stream_state_table)

        live_cfg = LiveInferenceConfig(
            artifact_bundle_path=self.cfg.artifact_bundle_path,
            resume_state_path=self.cfg.resume_state_path,
        )
        live_cfg.runtime.device = self.cfg.device
        live_cfg.runtime.outputs_dir = self.cfg.outputs_dir
        live_cfg.runtime.run_name = self.cfg.run_name
        live_cfg.runtime.save_arrays = self.cfg.save_arrays
        live_cfg.runtime.save_state_snapshot = self.cfg.save_state_snapshot

        live_cfg.eval.compute_unscaled_metrics = self.cfg.compute_unscaled_metrics

        live_cfg.adapt.mode = self.cfg.live_mode
        live_cfg.adapt.n_inner = self.cfg.n_inner
        live_cfg.adapt.learning_rate = self.cfg.adapt_learning_rate
        live_cfg.adapt.weight_decay = self.cfg.adapt_weight_decay
        live_cfg.adapt.grad_clip_norm = self.cfg.adapt_grad_clip_norm
        live_cfg.adapt.save_adapted_checkpoint = False
        live_cfg.adapt.use_amp = False
        live_cfg.adapt.require_batch_size_one = True

        self.service = LSTDLiveInferenceService(live_cfg)
        self.service.model.eval()
        self.service._set_online_trainable_mode(self.cfg.live_mode)

        self.gap_client = BinanceHistoricalKlinesClient(
            base_url=self.cfg.gap_fill_base_url,
            timeout=self.cfg.gap_fill_timeout,
            max_retries=self.cfg.gap_fill_max_retries,
        )

        telemetry_cfg = TelemetryConfig(
            enabled=self.cfg.telemetry_enabled,
            bucket=self.cfg.telemetry_bucket,
            prefix=self.cfg.telemetry_prefix,
            region=self.cfg.aws_region,
            local_dir=self.cfg.telemetry_local_dir,
        )
        self.telemetry = S3TelemetryWriter(
            telemetry_cfg,
            run_id=self.service.run_id,
            symbol=self.cfg.live_symbol,
            interval=self.cfg.live_interval,
        )

        self.warm_rows_processed = 0
        self.live_rows_processed = 0
        self.realized_windows = 0
        self.last_scaled_metrics: Optional[dict[str, Any]] = None
        self.last_unscaled_metrics: Optional[dict[str, Any]] = None

        self._write_status("starting")
        self._save_snapshot()
        self._write_summary()

    def run_forever(self) -> None:
        LOGGER.info("worker started run_id=%s symbol=%s interval=%s mode=%s", self.service.run_id, self.cfg.live_symbol, self.cfg.live_interval, self.cfg.live_mode)
        self.telemetry.log_event(
            "worker_started",
            {
                "artifact_bundle_path": self.cfg.artifact_bundle_path,
                "resume_state_path": self.cfg.resume_state_path or None,
                "mode": self.cfg.live_mode,
                "n_inner": self.cfg.n_inner,
            },
        )

        while not STOP_REQUESTED:
            message = self._receive_one(wait_seconds=self.cfg.poll_wait_seconds)
            if message is None:
                self._write_status("idle")
                self._write_summary()
                continue

            try:
                self._warm_until_caught_up(first_message=message)
            except Exception:
                LOGGER.exception("warmup/catchup failed")
                self._write_status("warmup_failed")
                raise

            self._arm_prediction_if_possible()
            self._write_status("live")
            self._write_summary()
            break

        while not STOP_REQUESTED:
            message = self._receive_one(wait_seconds=self.cfg.poll_wait_seconds)
            if message is None:
                self._write_status("live_idle")
                self._write_summary()
                continue

            candle = self._extract_candle_from_message(message)
            if candle is None:
                self._delete_message(message)
                continue

            current_open_time = int(candle["open_time"])
            last_completed = self.service.live_state.last_completed_open_time
            if last_completed is not None and current_open_time <= int(last_completed):
                LOGGER.info("dropping stale candle open_time=%s last_completed=%s", current_open_time, last_completed)
                self.telemetry.log_event(
                    "stale_message_dropped",
                    {"last_completed_open_time": int(last_completed)},
                    open_time=current_open_time,
                )
                self._delete_message(message)
                continue

            expected_next = None if last_completed is None else int(last_completed) + STEP_MS
            if expected_next is not None and current_open_time != expected_next:
                LOGGER.warning("gap detected in live mode expected_next=%s actual=%s; switching back to warm mode", expected_next, current_open_time)
                self.telemetry.log_event(
                    "live_gap_detected",
                    {
                        "expected_next_open_time": expected_next,
                        "actual_open_time": current_open_time,
                    },
                    open_time=current_open_time,
                )
                self.service.live_state.pending_predictions = []
                self._warm_until_caught_up(first_message=message)
                self._arm_prediction_if_possible()
                self._write_status("live")
                self._write_summary()
                continue

            self._process_live_message(message, candle)
            self._write_status("live")
            self._write_summary()

        self._write_status("stopped")
        self._write_summary()
        LOGGER.info("worker stopped")

    def _warm_until_caught_up(self, first_message: dict[str, Any]) -> None:
        message = first_message
        while message is not None and not STOP_REQUESTED:
            candle = self._extract_candle_from_message(message)
            if candle is None:
                self._delete_message(message)
                message = self._receive_one(wait_seconds=self.cfg.backlog_drain_wait_seconds)
                continue

            self._catch_up_before_open_time(int(candle["open_time"]))
            self._warm_single_candle(candle)
            self._delete_message(message)

            message = self._receive_one(wait_seconds=self.cfg.backlog_drain_wait_seconds)

    def _catch_up_before_open_time(self, incoming_open_time: int) -> None:
        last_completed = self.service.live_state.last_completed_open_time
        if last_completed is None:
            return

        start_open_time = int(last_completed) + STEP_MS
        end_open_time = int(incoming_open_time) - STEP_MS
        if end_open_time < start_open_time:
            return

        LOGGER.info("catching up REST gap start=%s end=%s", start_open_time, end_open_time)
        self.telemetry.log_event(
            "catchup_started",
            {
                "from_open_time": start_open_time,
                "to_open_time": end_open_time,
            },
            open_time=incoming_open_time,
        )

        gap_df = self.gap_client.fetch_historical_klines(
            symbol=self.cfg.live_symbol,
            interval=self.cfg.live_interval,
            start_ms=start_open_time,
            end_ms=end_open_time + STEP_MS,
            request_limit=self.cfg.gap_fill_request_limit,
            sleep_seconds=self.cfg.gap_fill_sleep_seconds,
        )

        if gap_df.empty:
            LOGGER.warning("REST catchup returned no rows start=%s end=%s", start_open_time, end_open_time)
            self.telemetry.log_event(
                "catchup_empty",
                {
                    "from_open_time": start_open_time,
                    "to_open_time": end_open_time,
                },
                open_time=incoming_open_time,
            )
            return

        self._advance_history_only(gap_df, reason="rest_gap")
        self.telemetry.log_event(
            "catchup_completed",
            {
                "rows": int(len(gap_df)),
                "from_open_time": start_open_time,
                "to_open_time": end_open_time,
            },
            open_time=end_open_time,
        )

    def _warm_single_candle(self, candle: dict[str, Any]) -> None:
        raw_df = self._single_candle_df(candle)
        self._advance_history_only(raw_df, reason="queue_warm")

    def _advance_history_only(self, raw_df: pd.DataFrame, *, reason: str) -> None:
        raw_df = self._normalize_raw_df(raw_df)
        if raw_df.empty:
            return

        feat_df, lstd_df = self.service._process_raw_into_lstd_rows(raw_df)
        if feat_df.empty or lstd_df.empty:
            return

        scaled_rows = self.service._scale_lstd_rows(lstd_df)
        rows_done = 0

        for i in range(len(lstd_df)):
            row_open_time = int(feat_df.iloc[i]["open_time"])
            row_date = str(lstd_df.iloc[i]["date"])
            scaled_row = np.asarray(scaled_rows[i], dtype=np.float32)
            self.service._append_history_row(scaled_row, row_date, row_open_time)
            rows_done += 1

        self.service.live_state.pending_predictions = []
        self.warm_rows_processed += rows_done
        self._save_snapshot()
        self._update_stream_state(reason=reason)

        latest_open_time = self.service.live_state.last_completed_open_time
        self.telemetry.log_event(
            "history_advanced",
            {
                "reason": reason,
                "rows_processed": rows_done,
                "warm_rows_processed_total": self.warm_rows_processed,
                "last_completed_open_time": latest_open_time,
            },
            open_time=latest_open_time,
        )

    def _arm_prediction_if_possible(self) -> None:
        if self.service.live_state.pending_predictions:
            return
        if len(self.service.live_state.history_scaled_rows) < self.service.seq_len:
            return

        self.service._enqueue_prediction()
        target_open_time = None
        if self.service.live_state.last_completed_open_time is not None:
            target_open_time = int(self.service.live_state.last_completed_open_time) + STEP_MS

        self.telemetry.log_event(
            "prediction_armed",
            {
                "pending_predictions": len(self.service.live_state.pending_predictions),
                "target_open_time": target_open_time,
            },
            open_time=target_open_time,
        )
        self._save_snapshot()
        self._update_stream_state(reason="prediction_armed")

    def _process_live_message(self, message: dict[str, Any], candle: dict[str, Any]) -> None:
        raw_df = self._single_candle_df(candle)
        raw_df = self._normalize_raw_df(raw_df)

        feat_df, lstd_df = self.service._process_raw_into_lstd_rows(raw_df)
        if feat_df.empty or lstd_df.empty:
            self._delete_message(message)
            return

        scaled_rows = self.service._scale_lstd_rows(lstd_df)

        for i in range(len(lstd_df)):
            row_open_time = int(feat_df.iloc[i]["open_time"])
            row_date = str(lstd_df.iloc[i]["date"])
            scaled_row = np.asarray(scaled_rows[i], dtype=np.float32)

            self.service._append_future_row_to_pending(scaled_row)
            self.service._append_history_row(scaled_row, row_date, row_open_time)
            self.live_rows_processed += 1

            while self.service.live_state.pending_predictions and self.service._pending_ready(self.service.live_state.pending_predictions[0]):
                pending = self.service.live_state.pending_predictions.pop(0)

                batch_x_np = np.asarray(pending.batch_x, dtype=np.float32)
                label_context_np = np.asarray(pending.label_context, dtype=np.float32)
                future_rows_np = np.asarray(pending.future_rows[: self.service.pred_len], dtype=np.float32)
                predicted_full = np.asarray(pending.predicted_seq, dtype=np.float32)

                pred_eval = self.service._slice_eval_channels(predicted_full)
                true_eval = self.service._slice_eval_channels(future_rows_np)

                scaled_metrics = regression_metrics(pred_eval, true_eval)
                unscaled_metrics = None
                inv = self.service._inverse_scale_for_eval(pred_eval, true_eval)
                if inv is not None:
                    preds_u, trues_u = inv
                    unscaled_metrics = regression_metrics(preds_u, trues_u)

                if self.cfg.live_mode != "none":
                    self.service._update_model_from_realized_target(
                        batch_x_np=batch_x_np,
                        label_context_np=label_context_np,
                        future_rows_np=future_rows_np,
                        mode=self.cfg.live_mode,
                        n_inner=self.cfg.n_inner,
                    )

                self.realized_windows += 1
                self.last_scaled_metrics = scaled_metrics
                self.last_unscaled_metrics = unscaled_metrics

                start_open_time = pending.issued_after_open_time
                first_target_open_time = None
                if start_open_time is not None:
                    first_target_open_time = int(start_open_time) + STEP_MS

                self.telemetry.log_event(
                    "prediction_realized",
                    {
                        "realized_windows_total": self.realized_windows,
                        "first_target_open_time": first_target_open_time,
                        "scaled_metrics": scaled_metrics,
                        "unscaled_metrics": unscaled_metrics,
                        "pred_shape": list(pred_eval.shape),
                        "true_shape": list(true_eval.shape),
                        "mode": self.cfg.live_mode,
                    },
                    open_time=first_target_open_time,
                )

            self.service._enqueue_prediction()

        self._save_snapshot()
        self._update_stream_state(reason="live_processed")
        self._delete_message(message)

    def _save_snapshot(self) -> Optional[str]:
        if not self.cfg.save_state_snapshot:
            return None
        return self.service._save_runtime_snapshot(self.cfg.state_snapshot_name)

    def _receive_one(self, *, wait_seconds: int) -> Optional[dict[str, Any]]:
        response = self.sqs.receive_message(
            QueueUrl=self.cfg.raw_queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=max(0, int(wait_seconds)),
            VisibilityTimeout=max(30, int(self.cfg.visibility_timeout_seconds)),
            AttributeNames=["All"],
            MessageAttributeNames=["All"],
        )
        messages = response.get("Messages", [])
        if not messages:
            return None
        return cast(dict[str, Any], messages[0])

    def _delete_message(self, message: dict[str, Any]) -> None:
        receipt_handle = message.get("ReceiptHandle")
        if not receipt_handle:
            return
        self.sqs.delete_message(
            QueueUrl=self.cfg.raw_queue_url,
            ReceiptHandle=receipt_handle,
        )

    def _extract_candle_from_message(self, message: dict[str, Any]) -> Optional[dict[str, Any]]:
        try:
            body = json.loads(str(message["Body"]))
        except Exception:
            LOGGER.exception("failed to parse message body as JSON")
            return None

        candle = body.get("candle", body)
        if not isinstance(candle, dict):
            return None

        symbol = str(candle.get("symbol", "")).upper().strip()
        interval = str(candle.get("interval", "")).strip()
        if symbol != self.cfg.live_symbol or interval != self.cfg.live_interval:
            return None

        required = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
        for key in required:
            if key not in candle:
                raise ValueError(f"SQS candle message missing required field: {key}")

        return {
            "symbol": symbol,
            "interval": interval,
            "open_time": int(candle["open_time"]),
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle["volume"]),
            "close_time": int(candle["close_time"]),
            "quote_asset_volume": float(candle["quote_asset_volume"]),
            "number_of_trades": int(candle["number_of_trades"]),
            "taker_buy_base_asset_volume": float(candle["taker_buy_base_asset_volume"]),
            "taker_buy_quote_asset_volume": float(candle["taker_buy_quote_asset_volume"]),
            "ignore": int(candle.get("ignore", 0)),
        }

    def _single_candle_df(self, candle: dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([candle])
        return self._normalize_raw_df(df)

    def _normalize_raw_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return raw_df.copy()

        df = raw_df.copy()
        if "open_time" not in df.columns:
            raise ValueError("raw dataframe must contain open_time")
        if "close_time" not in df.columns:
            df["close_time"] = df["open_time"].astype("int64") + STEP_MS - 1
        if "ignore" not in df.columns:
            df["ignore"] = 0

        if "open_dt" not in df.columns:
            df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        else:
            df["open_dt"] = pd.to_datetime(df["open_dt"], utc=True, errors="coerce")

        if "close_dt" not in df.columns:
            df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        else:
            df["close_dt"] = pd.to_datetime(df["close_dt"], utc=True, errors="coerce")

        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        return df

    def _write_status(self, status: str) -> None:
        self._update_stream_state(reason=status)

    def _write_summary(self) -> None:
        payload = {
            "status": "stopping" if STOP_REQUESTED else "running",
            "mode": self.cfg.live_mode,
            "warm_rows_processed": self.warm_rows_processed,
            "live_rows_processed": self.live_rows_processed,
            "realized_windows": self.realized_windows,
            "pending_predictions": len(self.service.live_state.pending_predictions),
            "last_completed_open_time": self.service.live_state.last_completed_open_time,
            "last_scaled_metrics": self.last_scaled_metrics,
            "last_unscaled_metrics": self.last_unscaled_metrics,
            "out_dir": self.service.out_dir,
        }
        self.telemetry.write_latest_summary(payload)

    def _update_stream_state(self, *, reason: str) -> None:
        if self.ddb_table is None:
            return

        now_epoch = int(time.time())
        values: dict[str, Any] = {
            ":status": reason,
            ":run_id": self.service.run_id,
            ":artifact": self.cfg.artifact_bundle_path,
            ":updated_at": now_epoch,
            ":last_completed": (
                None
                if self.service.live_state.last_completed_open_time is None
                else int(self.service.live_state.last_completed_open_time)
            ),
            ":realized_windows": int(self.realized_windows),
            ":warm_rows": int(self.warm_rows_processed),
            ":live_rows": int(self.live_rows_processed),
        }

        self.ddb_table.update_item(
            Key={"symbol": self.cfg.live_symbol, "interval": self.cfg.live_interval},
            UpdateExpression=(
                "SET inference_status = :status, "
                "inference_run_id = :run_id, "
                "inference_artifact_path = :artifact, "
                "inference_last_completed_open_time = :last_completed, "
                "inference_realized_windows = :realized_windows, "
                "inference_warm_rows_processed = :warm_rows, "
                "inference_live_rows_processed = :live_rows, "
                "inference_updated_at = :updated_at"
            ),
            ExpressionAttributeValues=values,
        )


def _env_required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.info("stop requested")


def _configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    _configure_logging()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    cfg = WorkerConfig.from_env()
    worker = LSTDSQSWorker(cfg)
    worker.run_forever()


if __name__ == "__main__":
    main()
