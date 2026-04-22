# orchestration/local_worker_replay.py

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import pandas as pd

from binanace_history.client import BinanceHistoricalKlinesClient
from live_inference.config import LiveInferenceConfig
from live_inference.service import LSTDLiveInferenceService
from training.metrics import regression_metrics
from training.utils import save_json


STEP_MS = 60_000


class LocalWorkerReplay:
    """
    Single-machine harness that mimics the intended sqs_worker behavior.

    Flow:
    1. Restore training artifact state.
    2. Fill any gap from artifact.last_completed_open_time to the first replay candle.
    3. Optionally consume an initial backlog in warm-only mode.
    4. Only then arm prediction.
    5. Process remaining rows as real live candles.

    This is the cleanest way to validate your AWS worker logic locally on one GPU box
    before you wire SQS, DynamoDB, and collector EC2 around it.
    """

    def __init__(
        self,
        live_cfg: LiveInferenceConfig,
        *,
        raw_csv_path: str,
        symbol: str,
        interval: str,
        initial_queue_backlog_bars: int = 0,
    ) -> None:
        self.live_cfg = deepcopy(live_cfg)
        self.raw_csv_path = raw_csv_path
        self.symbol = symbol
        self.interval = interval
        self.initial_queue_backlog_bars = int(initial_queue_backlog_bars)

        if self.initial_queue_backlog_bars < 0:
            raise ValueError("initial_queue_backlog_bars must be >= 0")

        self.service = LSTDLiveInferenceService(self.live_cfg)
        self.service.model.eval()
        self.service._set_online_trainable_mode(self.live_cfg.adapt.mode)

        self.rest_client = BinanceHistoricalKlinesClient(
            base_url=self.live_cfg.gap_fill.base_url,
            timeout=self.live_cfg.gap_fill.timeout,
            max_retries=self.live_cfg.gap_fill.max_retries,
        )

        self.gap_rows_filled = 0
        self.backlog_rows_warmed = 0
        self.live_rows_processed = 0
        self.realized_windows = 0
        self.last_scaled_metrics: Optional[dict[str, Any]] = None
        self.last_unscaled_metrics: Optional[dict[str, Any]] = None

    def run(self) -> dict[str, Any]:
        raw_df = pd.read_csv(self.raw_csv_path)
        raw_df = self._normalize_raw_df(raw_df)
        if raw_df.empty:
            raise RuntimeError("Local worker replay input CSV is empty")

        first_open_time = int(raw_df.iloc[0]["open_time"])
        self._catch_up_before_open_time(first_open_time)

        backlog_n = min(self.initial_queue_backlog_bars, len(raw_df))
        if backlog_n > 0:
            backlog_df = raw_df.iloc[:backlog_n].copy().reset_index(drop=True)
            self._advance_history_only(backlog_df)
            self.backlog_rows_warmed += len(backlog_df)
            live_df = raw_df.iloc[backlog_n:].copy().reset_index(drop=True)
        else:
            live_df = raw_df

        self._arm_prediction_if_possible()

        for i in range(len(live_df)):
            single_row_df = live_df.iloc[[i]].copy().reset_index(drop=True)
            self._process_live_row(single_row_df)

        summary = {
            "run_id": self.service.run_id,
            "evaluation_protocol": "local_worker_replay",
            "artifact_bundle_path": self.live_cfg.artifact_bundle_path,
            "resume_state_path": self.live_cfg.resume_state_path or None,
            "raw_csv_path": self.raw_csv_path,
            "symbol": self.symbol,
            "interval": self.interval,
            "online_mode": self.live_cfg.adapt.mode,
            "n_inner": self.live_cfg.adapt.n_inner,
            "gap_rows_filled": self.gap_rows_filled,
            "backlog_rows_warmed": self.backlog_rows_warmed,
            "live_rows_processed": self.live_rows_processed,
            "realized_windows": self.realized_windows,
            "last_completed_open_time": self.service.live_state.last_completed_open_time,
            "last_scaled_metrics": self.last_scaled_metrics,
            "last_unscaled_metrics": self.last_unscaled_metrics,
            "out_dir": self.service.out_dir,
        }

        save_json(os.path.join(self.service.out_dir, "local_worker_replay_summary.json"), summary)
        return summary

    def _catch_up_before_open_time(self, incoming_open_time: int) -> None:
        last_completed = self.service.live_state.last_completed_open_time
        if last_completed is None:
            return

        start_open_time = int(last_completed) + STEP_MS
        end_open_time = int(incoming_open_time) - STEP_MS
        if end_open_time < start_open_time:
            return

        gap_df = self.rest_client.fetch_historical_klines(
            symbol=self.symbol,
            interval=self.interval,
            start_ms=start_open_time,
            end_ms=end_open_time + STEP_MS,
            request_limit=self.live_cfg.gap_fill.request_limit,
            sleep_seconds=self.live_cfg.gap_fill.sleep_seconds,
        )
        gap_df = self._normalize_raw_df(gap_df)
        if gap_df.empty:
            return

        self._advance_history_only(gap_df)
        self.gap_rows_filled += len(gap_df)

    def _advance_history_only(self, raw_df: pd.DataFrame) -> None:
        feat_df, lstd_df = self.service._process_raw_into_lstd_rows(raw_df)
        if feat_df.empty or lstd_df.empty:
            return

        scaled_rows = self.service._scale_lstd_rows(lstd_df)
        for i in range(len(lstd_df)):
            row_open_time = int(feat_df.iloc[i]["open_time"])
            row_date = str(lstd_df.iloc[i]["date"])
            scaled_row = np.asarray(scaled_rows[i], dtype=np.float32)
            self.service._append_history_row(scaled_row, row_date, row_open_time)

        self.service.live_state.pending_predictions = []

    def _arm_prediction_if_possible(self) -> None:
        if self.service.live_state.pending_predictions:
            return
        if len(self.service.live_state.history_scaled_rows) < self.service.seq_len:
            return
        self.service._enqueue_prediction()

    def _process_live_row(self, raw_df: pd.DataFrame) -> None:
        feat_df, lstd_df = self.service._process_raw_into_lstd_rows(raw_df)
        if feat_df.empty or lstd_df.empty:
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
                self.last_scaled_metrics = scaled_metrics

                inv = self.service._inverse_scale_for_eval(pred_eval, true_eval)
                if inv is not None:
                    preds_u, trues_u = inv
                    self.last_unscaled_metrics = regression_metrics(preds_u, trues_u)
                else:
                    self.last_unscaled_metrics = None

                if self.live_cfg.adapt.mode != "none":
                    self.service._update_model_from_realized_target(
                        batch_x_np=batch_x_np,
                        label_context_np=label_context_np,
                        future_rows_np=future_rows_np,
                        mode=self.live_cfg.adapt.mode,
                        n_inner=self.live_cfg.adapt.n_inner,
                    )

                self.realized_windows += 1

            self.service._enqueue_prediction()

    @staticmethod
    def _normalize_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return raw_df.copy()

        df = raw_df.copy()
        if "open_time" not in df.columns:
            raise ValueError("raw dataframe must contain 'open_time'")

        if "close_time" not in df.columns:
            df["close_time"] = df["open_time"].astype("int64") + STEP_MS - 1
        if "ignore" not in df.columns:
            df["ignore"] = 0

        if "open_dt" in df.columns:
            df["open_dt"] = pd.to_datetime(df["open_dt"], utc=True, errors="coerce")
        else:
            df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

        if "close_dt" in df.columns:
            df["close_dt"] = pd.to_datetime(df["close_dt"], utc=True, errors="coerce")
        else:
            df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        return df
