# live_inference/service.py

from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from binanace_history.client import BinanceHistoricalKlinesClient
from datasets.scalers import StandardScalerNumpy
from datasets.utils import interval_to_millis
from feature_engineering.config import FeatureConfig, LSTDExportConfig
from feature_engineering.export import build_lstd_ready_frame
from feature_engineering.state import FeaturePipelineState, StatefulFeatureEngineer
from lstd_core.config import LSTDModelConfig
from lstd_core.model import LSTDNet
from training.metrics import regression_metrics
from training.utils import choose_device, ensure_dir, save_json, timestamp_tag

from .config import LiveInferenceConfig
from .state import LiveRuntimeState, PendingPredictionState


class LSTDLiveInferenceService:
    """
    Online / live inference service.

    Key properties:
    - starts only from a training artifact bundle or a saved live-state snapshot
    - does not depend on LSTDDataset or prebuilt training datasets
    - restores model state, optimizer state, scaler state, and rolling feature state
    - keeps a resumable live runtime state for AWS-style online learning
    """
    @staticmethod
    def _to_float_row(arr: np.ndarray) -> list[float]:
        row = np.asarray(arr, dtype=np.float32).reshape(-1)
        return [float(x) for x in row]

    @staticmethod
    def _to_float_matrix(arr: np.ndarray) -> list[list[float]]:
        mat = np.asarray(arr, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {mat.shape}")
        return [[float(x) for x in row] for row in mat]

    def __init__(self, cfg: LiveInferenceConfig):
        self.cfg = deepcopy(cfg)
        if not self.cfg.artifact_bundle_path:
            raise ValueError("LiveInferenceConfig.artifact_bundle_path is required")

        self.device = choose_device(self.cfg.runtime.device)
        self.bundle = torch.load(self.cfg.artifact_bundle_path, map_location="cpu")

        self.window_cfg = dict(self.bundle["window_config"])
        self.model_cfg = LSTDModelConfig(**self.bundle["model_config"])
        self.feature_cfg = FeatureConfig(**self.bundle["feature_config"])
        self.export_cfg = LSTDExportConfig(**self.bundle["export_config"])

        self.seq_len = int(self.window_cfg["seq_len"])
        self.label_len = int(self.window_cfg["label_len"])
        self.pred_len = int(self.window_cfg["pred_len"])
        self.features_mode = str(self.window_cfg["features"]).upper()
        self.target_column = str(self.bundle["target_column"])
        self.model_columns = list(self.bundle["model_columns"])
        self.enc_in = len(self.model_columns)

        self.input_scaler = StandardScalerNumpy.from_state(self.bundle["input_scaler_state"])
        self.date_scaler = None
        if self.bundle.get("date_scaler_state") is not None:
            self.date_scaler = StandardScalerNumpy.from_state(self.bundle["date_scaler_state"])

        self.feature_engineer = StatefulFeatureEngineer(self.feature_cfg)

        self.model = LSTDNet(self.model_cfg, device=self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.adapt.learning_rate,
            weight_decay=self.cfg.adapt.weight_decay,
        )

        self.use_amp = bool(self.cfg.adapt.use_amp) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.live_state = LiveRuntimeState.from_dict(self.bundle["initial_live_state"])
        self._restore_from_bundle_or_snapshot()

        self.run_id = f"{self.cfg.runtime.run_name}_{timestamp_tag()}"
        self.out_dir = os.path.join(self.cfg.runtime.outputs_dir, self.run_id)
        ensure_dir(self.out_dir)

    def _restore_from_bundle_or_snapshot(self) -> None:
        if self.cfg.resume_state_path:
            snapshot = torch.load(self.cfg.resume_state_path, map_location="cpu")
            self.model.load_state_dict(snapshot["model_state_dict"])
            self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            if self.use_amp and snapshot.get("amp_scaler_state_dict") is not None:
                self.scaler.load_state_dict(snapshot["amp_scaler_state_dict"])
            self.live_state = LiveRuntimeState.from_dict(snapshot["live_state"])
            return

        self.model.load_state_dict(self.bundle["model_state_dict"])
        if self.bundle.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(self.bundle["optimizer_state_dict"])
        if self.use_amp and self.bundle.get("amp_scaler_state_dict") is not None:
            self.scaler.load_state_dict(self.bundle["amp_scaler_state_dict"])

    def _load_live_raw_df(self, raw_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if raw_df is not None:
            df = raw_df.copy()
        else:
            if not self.cfg.live_root_path or not self.cfg.live_data_path:
                raise ValueError("Provide raw_df directly or set live_root_path + live_data_path.")
            df = pd.read_csv(os.path.join(self.cfg.live_root_path, self.cfg.live_data_path))

        if df.empty:
            return df
        if "open_time" not in df.columns:
            raise ValueError("Live raw dataframe must contain 'open_time'.")

        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        return df

    def _maybe_fill_gap(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return raw_df

        feature_state = FeaturePipelineState.from_dict(self.live_state.feature_pipeline_state)
        if feature_state.last_open_time is None:
            return raw_df

        if not self.cfg.gap_fill.interval:
            return raw_df

        step_ms = interval_to_millis(self.cfg.gap_fill.interval)
        expected_next_open = int(feature_state.last_open_time) + step_ms
        first_live_open = int(raw_df.iloc[0]["open_time"])

        if first_live_open <= feature_state.last_open_time:
            return raw_df[raw_df["open_time"] > feature_state.last_open_time].copy().reset_index(drop=True)

        if first_live_open == expected_next_open:
            return raw_df

        if not self.cfg.gap_fill.enabled:
            raise RuntimeError(
                "Gap detected between training/live state and incoming raw data. "
                "Enable gap_fill or provide a bridge dataframe before starting live inference."
            )

        client = BinanceHistoricalKlinesClient(
            base_url=self.cfg.gap_fill.base_url,
            timeout=self.cfg.gap_fill.timeout,
            max_retries=self.cfg.gap_fill.max_retries,
        )
        gap_df = client.fetch_historical_klines(
            symbol=self.cfg.gap_fill.symbol,
            interval=self.cfg.gap_fill.interval,
            start_ms=expected_next_open,
            end_ms=first_live_open,
            request_limit=self.cfg.gap_fill.request_limit,
            sleep_seconds=self.cfg.gap_fill.sleep_seconds,
        )

        out = pd.concat([gap_df, raw_df], ignore_index=True)
        out = out.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        return out

    def _slice_eval_channels(self, arr: np.ndarray) -> np.ndarray:
        if self.features_mode == "MS":
            return arr[:, -1:]
        return arr

    def _inverse_scale_for_eval(
        self,
        pred_np: np.ndarray,
        true_np: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not bool(self.window_cfg.get("scale", True)):
            return None

        mean = np.asarray(self.input_scaler.mean_, dtype=np.float32)
        std = np.asarray(self.input_scaler.std_, dtype=np.float32)
        if mean is None or std is None:
            return None

        if self.features_mode in {"M", "S"}:
            return (
                pred_np * std.reshape(1, -1) + mean.reshape(1, -1),
                true_np * std.reshape(1, -1) + mean.reshape(1, -1),
            )

        target_mean = float(mean[0, -1])
        target_std = float(std[0, -1])
        return pred_np * target_std + target_mean, true_np * target_std + target_mean

    def _freeze_encoders(self) -> None:
        for module in [self.model.long_encoder, self.model.short_encoder]:
            for param in module.parameters():
                param.requires_grad_(False)

    def _set_online_trainable_mode(self, mode: str) -> None:
        mode_l = mode.lower()
        if mode_l not in {"none", "full", "regressor"}:
            raise ValueError("online mode must be one of: none, full, regressor")

        for param in self.model.parameters():
            param.requires_grad_(True)

        if mode_l == "regressor":
            self._freeze_encoders()

    def _scale_lstd_rows(self, lstd_df: pd.DataFrame) -> np.ndarray:
        values = lstd_df[self.model_columns].to_numpy(dtype=np.float32, copy=True)
        return self.input_scaler.transform(values)

    def _enqueue_prediction(self) -> None:
        if len(self.live_state.history_scaled_rows) < self.seq_len:
            return

        batch_x_np = np.asarray(self.live_state.history_scaled_rows[-self.seq_len:], dtype=np.float32)
        label_context_np = batch_x_np[-self.label_len:].copy()

        with torch.no_grad():
            x_tensor = torch.from_numpy(batch_x_np[None, :, :]).to(self.device)
            _, outputs_flat, _ = self.model(
                x_tensor,
                sample_latents=False,
                include_kl=False,
            )
            predicted_seq = outputs_flat.view(1, self.pred_len, self.enc_in).detach().cpu().numpy()[0]

        issued_after_open_time = self.live_state.last_completed_open_time
        self.live_state.pending_predictions.append(
            PendingPredictionState(
                batch_x=self._to_float_matrix(batch_x_np),
                label_context=self._to_float_matrix(label_context_np),
                predicted_seq=self._to_float_matrix(predicted_seq),
                future_rows=[],
                issued_after_open_time=issued_after_open_time,
            )
        )

    def _append_future_row_to_pending(self, scaled_row: np.ndarray) -> None:
        row_list = self._to_float_row(scaled_row)
        for pending in self.live_state.pending_predictions:
            pending.future_rows.append(row_list)

    def _append_history_row(self, scaled_row: np.ndarray, row_date: str, open_time: int) -> None:
        self.live_state.history_scaled_rows.append(self._to_float_row(scaled_row))
        self.live_state.history_dates.append(row_date)

        if len(self.live_state.history_scaled_rows) > self.seq_len:
            self.live_state.history_scaled_rows = self.live_state.history_scaled_rows[-self.seq_len:]
            self.live_state.history_dates = self.live_state.history_dates[-self.seq_len:]

        self.live_state.last_completed_open_time = int(open_time)

    def _pending_ready(self, pending: PendingPredictionState) -> bool:
        return len(pending.future_rows) >= self.pred_len

    def _update_model_from_realized_target(
        self,
        batch_x_np: np.ndarray,
        label_context_np: np.ndarray,
        future_rows_np: np.ndarray,
        mode: str,
        n_inner: int,
    ) -> None:
        if mode == "none":
            return

        batch_y_np = np.concatenate([label_context_np, future_rows_np], axis=0)
        batch_x = torch.from_numpy(batch_x_np[None, :, :])
        batch_y = torch.from_numpy(batch_y_np[None, :, :])

        for _ in range(int(n_inner)):
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    out = self._forward_losses(
                        batch_x,
                        batch_y,
                        sample_latents=True,
                        include_kl=True,
                    )

                self.scaler.scale(out["total_loss"]).backward()
                if self.cfg.adapt.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.adapt.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self._forward_losses(
                    batch_x,
                    batch_y,
                    sample_latents=True,
                    include_kl=True,
                )
                out["total_loss"].backward()
                if self.cfg.adapt.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.adapt.grad_clip_norm)
                self.optimizer.step()

            self.model.store_grad()

    def _forward_losses(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        *,
        sample_latents: bool,
        include_kl: bool,
    ) -> Dict[str, torch.Tensor]:
        x = batch_x.to(self.device).float()
        x_rec, outputs_flat, other_loss = self.model(
            x,
            sample_latents=sample_latents,
            include_kl=include_kl,
        )

        pred_flat, true_flat = self._extract_pred_true(outputs_flat, batch_y)
        pred_loss = self.criterion(pred_flat, true_flat)
        rec_loss = self.criterion(x_rec, x)
        total_loss = pred_loss + rec_loss + other_loss

        return {
            "x": x,
            "x_rec": x_rec,
            "outputs_flat": outputs_flat,
            "pred_flat": pred_flat,
            "true_flat": true_flat,
            "pred_loss": pred_loss,
            "rec_loss": rec_loss,
            "other_loss": other_loss,
            "total_loss": total_loss,
        }

    def _extract_pred_true(
        self,
        outputs_flat: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = outputs_flat.shape[0]
        pred_seq = outputs_flat.view(batch_size, self.pred_len, self.enc_in)
        true_seq_all = batch_y[:, -self.pred_len:, :].to(self.device).float()

        if self.features_mode == "MS":
            pred_seq = pred_seq[:, :, -1:]
            true_seq = true_seq_all[:, :, -1:]
        else:
            true_seq = true_seq_all

        return pred_seq.reshape(batch_size, -1), true_seq.reshape(batch_size, -1)

    def _process_raw_into_lstd_rows(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        current_state = FeaturePipelineState.from_dict(self.live_state.feature_pipeline_state)
        feat_df, new_state = self.feature_engineer.transform_incremental(raw_df, current_state)
        self.live_state.feature_pipeline_state = new_state.to_dict()

        if feat_df.empty:
            return feat_df, pd.DataFrame()

        lstd_df, _ = build_lstd_ready_frame(feat_df, self.export_cfg)
        return feat_df.reset_index(drop=True), lstd_df.reset_index(drop=True)

    def _capture_runtime_snapshot(self) -> dict[str, Any]:
        amp_state = self.scaler.state_dict() if self.use_amp else None
        return {
            "model_state_dict": deepcopy(self.model.state_dict()),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
            "amp_scaler_state_dict": deepcopy(amp_state),
            "live_state": deepcopy(self.live_state.to_dict()),
        }

    def _restore_runtime_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.model.load_state_dict(snapshot["model_state_dict"])
        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        if self.use_amp and snapshot.get("amp_scaler_state_dict") is not None:
            self.scaler.load_state_dict(snapshot["amp_scaler_state_dict"])
        self.live_state = LiveRuntimeState.from_dict(snapshot["live_state"])

    def _save_runtime_snapshot(self, filename: str) -> str:
        payload = self._capture_runtime_snapshot()
        path = os.path.join(self.out_dir, filename)
        torch.save(payload, path)
        return path

    def _run_stream(
        self,
        *,
        raw_df: Optional[pd.DataFrame],
        online_mode: str,
        n_inner: int,
        persist_state: bool,
        summary_name: str,
        arrays_prefix: str,
    ) -> Dict[str, Any]:
        mode = online_mode.lower()
        if mode not in {"none", "full", "regressor"}:
            raise ValueError("mode must be one of: none, full, regressor")
        if n_inner < 1:
            raise ValueError("n_inner must be >= 1")

        if self.cfg.adapt.require_batch_size_one and self.pred_len < 1:
            raise ValueError("pred_len must be >= 1")

        raw_input_df = self._load_live_raw_df(raw_df)
        raw_input_df = self._maybe_fill_gap(raw_input_df)
        feat_df, lstd_df = self._process_raw_into_lstd_rows(raw_input_df)
        if feat_df.empty or lstd_df.empty:
            raise RuntimeError("No new live rows were available after state/gap filtering.")

        scaled_rows = self._scale_lstd_rows(lstd_df)

        self.model.eval()
        self._set_online_trainable_mode(mode)

        if not self.live_state.pending_predictions and len(self.live_state.history_scaled_rows) >= self.seq_len:
            self._enqueue_prediction()

        pred_batches: List[np.ndarray] = []
        true_batches: List[np.ndarray] = []
        pred_losses: List[float] = []
        rec_losses: List[float] = []
        other_losses: List[float] = []

        t0 = time.time()
        pbar = tqdm(range(len(lstd_df)), total=len(lstd_df), desc=summary_name, dynamic_ncols=True)

        for i in pbar:
            row_open_time = int(feat_df.iloc[i]["open_time"])
            row_date = str(lstd_df.iloc[i]["date"])
            scaled_row = np.asarray(scaled_rows[i], dtype=np.float32)

            self._append_future_row_to_pending(scaled_row)
            self._append_history_row(scaled_row, row_date, row_open_time)

            while self.live_state.pending_predictions and self._pending_ready(self.live_state.pending_predictions[0]):
                pending = self.live_state.pending_predictions.pop(0)

                batch_x_np = np.asarray(pending.batch_x, dtype=np.float32)
                label_context_np = np.asarray(pending.label_context, dtype=np.float32)
                future_rows_np = np.asarray(pending.future_rows[:self.pred_len], dtype=np.float32)
                predicted_full = np.asarray(pending.predicted_seq, dtype=np.float32)

                pred_eval = self._slice_eval_channels(predicted_full)
                true_eval = self._slice_eval_channels(future_rows_np)

                pred_batches.append(pred_eval.astype(np.float32))
                true_batches.append(true_eval.astype(np.float32))

                pred_loss = float(np.mean((pred_eval - true_eval) ** 2))
                rec_losses.append(0.0)
                other_losses.append(0.0)
                pred_losses.append(pred_loss)

                self._update_model_from_realized_target(
                    batch_x_np=batch_x_np,
                    label_context_np=label_context_np,
                    future_rows_np=future_rows_np,
                    mode=mode,
                    n_inner=n_inner,
                )

            self._enqueue_prediction()

            if pred_batches:
                preds_so_far = np.concatenate(pred_batches, axis=0)
                trues_so_far = np.concatenate(true_batches, axis=0)
                metrics_so_far = regression_metrics(preds_so_far, trues_so_far)
                pbar.set_postfix(
                    realized=f"{preds_so_far.shape[0]}",
                    mse=f"{metrics_so_far['mse']:.5f}",
                    mae=f"{metrics_so_far['mae']:.5f}",
                    pred=f"{np.mean(pred_losses):.5f}",
                )

        if not pred_batches:
            raise RuntimeError("No realized prediction windows were produced from the live stream.")

        preds = np.concatenate(pred_batches, axis=0)
        trues = np.concatenate(true_batches, axis=0)
        scaled_metrics = regression_metrics(preds, trues)

        unscaled_metrics = None
        inv = None
        if self.cfg.eval.compute_unscaled_metrics:
            inv = self._inverse_scale_for_eval(preds, trues)
            if inv is not None:
                preds_u, trues_u = inv
                unscaled_metrics = regression_metrics(preds_u, trues_u)

        elapsed = time.time() - t0
        num_windows = int(preds.shape[0])
        result: Dict[str, Any] = {
            "run_id": self.run_id,
            "artifact_bundle_path": self.cfg.artifact_bundle_path,
            "resume_state_path": self.cfg.resume_state_path or None,
            "evaluation_protocol": (
                "online_replay_predict_then_update" if mode != "none" else "static_no_online_updates"
            ),
            "online_mode": mode,
            "n_inner": n_inner,
            "num_realized_windows": int(preds.shape[0]),
            "num_windows": num_windows,
            "pred_shape": list(preds.shape),
            "true_shape": list(trues.shape),
            "avg_pred_loss": float(np.mean(pred_losses)) if pred_losses else None,
            "avg_recon_loss": float(np.mean(rec_losses)) if rec_losses else None,
            "avg_other_loss": float(np.mean(other_losses)) if other_losses else None,
            "scaled_metrics": scaled_metrics,
            "unscaled_metrics": unscaled_metrics,
            "elapsed_seconds": float(elapsed),
        }

        if self.cfg.runtime.save_arrays:
            np.save(os.path.join(self.out_dir, f"{arrays_prefix}_preds.npy"), preds)
            np.save(os.path.join(self.out_dir, f"{arrays_prefix}_trues.npy"), trues)
            if inv is not None:
                preds_u, trues_u = inv
                np.save(os.path.join(self.out_dir, f"{arrays_prefix}_preds_unscaled.npy"), preds_u)
                np.save(os.path.join(self.out_dir, f"{arrays_prefix}_trues_unscaled.npy"), trues_u)

        if mode != "none" and self.cfg.adapt.save_adapted_checkpoint:
            adapted_ckpt_path = os.path.join(self.out_dir, f"{arrays_prefix}_adapted_model.pth")
            torch.save(self.model.state_dict(), adapted_ckpt_path)
            result["adapted_checkpoint"] = adapted_ckpt_path

        if persist_state and self.cfg.runtime.save_state_snapshot:
            live_state_path = self._save_runtime_snapshot(f"{arrays_prefix}_live_state.pt")
            result["live_state_path"] = live_state_path

        save_json(os.path.join(self.out_dir, f"{arrays_prefix}_summary.json"), result)
        return result

    def run_online(
        self,
        raw_df: Optional[pd.DataFrame] = None,
        mode: Optional[str] = None,
        n_inner: Optional[int] = None,
    ) -> Dict[str, Any]:
        online_mode = (mode or self.cfg.adapt.mode).lower()
        inner_steps = int(self.cfg.adapt.n_inner if n_inner is None else n_inner)
        return self._run_stream(
            raw_df=raw_df,
            online_mode=online_mode,
            n_inner=inner_steps,
            persist_state=True,
            summary_name=f"Online-live[{online_mode}]",
            arrays_prefix=f"live_{online_mode}",
        )

    def run_static(self, raw_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        snapshot = self._capture_runtime_snapshot()
        try:
            return self._run_stream(
                raw_df=raw_df,
                online_mode="none",
                n_inner=1,
                persist_state=False,
                summary_name="Static-live",
                arrays_prefix="live_static",
            )
        finally:
            self._restore_runtime_snapshot(snapshot)
