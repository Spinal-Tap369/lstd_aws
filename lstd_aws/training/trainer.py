# training/trainer.py

from __future__ import annotations

import os
import time
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from lstd_core.model import LSTDNet

from .config import FitTrainConfig
from .data import TrainValLoaderBundle, build_train_val_loaders
from .utils import ensure_dir, save_json, set_seed, choose_device, timestamp_tag


class LSTDFitTrainer:
    """
    Offline fitting only.

    Important differences versus your previous trainer:
    - consumes RAW historical csv input
    - splits raw data BEFORE feature engineering
    - fits train-only scalers explicitly
    - publishes a resumable artifact bundle for live inference
    """

    def __init__(self, cfg: FitTrainConfig):
        self.cfg = deepcopy(cfg)

        set_seed(self.cfg.runtime.seed)
        self.device = choose_device(self.cfg.runtime.device)

        self.bundle: TrainValLoaderBundle = build_train_val_loaders(self.cfg)

        self.cfg.model.seq_len = self.cfg.windows.seq_len
        self.cfg.model.pred_len = self.cfg.windows.pred_len
        self.cfg.model.enc_in = self.bundle.enc_in

        self.model = LSTDNet(self.cfg.model, device=self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.optim.learning_rate,
            weight_decay=self.cfg.optim.weight_decay,
        )

        self.use_amp = bool(self.cfg.optim.use_amp) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.run_id = f"{self.cfg.runtime.experiment_name}_{timestamp_tag()}"
        self.ckpt_dir = os.path.join(self.cfg.runtime.checkpoints_dir, self.run_id)
        self.out_dir = os.path.join(self.cfg.runtime.outputs_dir, self.run_id)
        ensure_dir(self.ckpt_dir)
        ensure_dir(self.out_dir)

        self.best_ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pth")
        self.artifact_bundle_path = os.path.join(self.out_dir, self.cfg.runtime.artifact_bundle_name)

    def _extract_pred_true(
        self,
        outputs_flat: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_len = self.cfg.windows.pred_len
        enc_in = self.bundle.enc_in
        batch_size = outputs_flat.shape[0]

        pred_seq = outputs_flat.view(batch_size, pred_len, enc_in)
        true_seq_all = batch_y[:, -pred_len:, :].to(self.device).float()

        if self.cfg.windows.features.upper() == "MS":
            pred_seq = pred_seq[:, :, -1:]
            true_seq = true_seq_all[:, :, -1:]
        else:
            true_seq = true_seq_all

        pred_flat = pred_seq.reshape(batch_size, -1)
        true_flat = true_seq.reshape(batch_size, -1)
        return pred_flat, true_flat

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

    def _train_one_epoch(self, epoch_idx: int) -> float:
        self.model.train()
        losses: List[float] = []

        pbar = tqdm(
            self.bundle.train_loader,
            total=len(self.bundle.train_loader),
            desc=f"Train {epoch_idx + 1}/{self.cfg.optim.train_epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for step_idx, batch in enumerate(pbar, start=1):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            del batch_x_mark, batch_y_mark

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

                if self.cfg.optim.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)

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

                if self.cfg.optim.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)

                self.optimizer.step()

            self.model.store_grad()

            total_loss_val = float(out["total_loss"].detach().cpu().item())
            pred_loss_val = float(out["pred_loss"].detach().cpu().item())
            rec_loss_val = float(out["rec_loss"].detach().cpu().item())
            other_loss_val = float(out["other_loss"].detach().cpu().item())
            losses.append(total_loss_val)

            pbar.set_postfix(
                step=f"{step_idx}/{len(self.bundle.train_loader)}",
                loss=f"{total_loss_val:.4f}",
                pred=f"{pred_loss_val:.4f}",
                rec=f"{rec_loss_val:.4f}",
                oth=f"{other_loss_val:.4f}",
                avg=f"{np.mean(losses):.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        return float(np.mean(losses)) if losses else float("inf")

    @torch.no_grad()
    def _validate_static(self) -> Optional[float]:
        if self.bundle.val_loader is None:
            return None

        self.model.eval()
        losses: List[float] = []

        pbar = tqdm(
            self.bundle.val_loader,
            total=len(self.bundle.val_loader),
            desc="Validate",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            del batch_x_mark, batch_y_mark

            out = self._forward_losses(
                batch_x,
                batch_y,
                sample_latents=False,
                include_kl=False,
            )

            loss = float(out["pred_loss"].detach().cpu().item())
            losses.append(loss)
            pbar.set_postfix(avg_pred=f"{np.mean(losses):.4f}")

        self.model.train()
        return float(np.mean(losses)) if losses else None

    def _capture_training_state(self) -> dict[str, Any]:
        amp_state = self.scaler.state_dict() if self.use_amp else None
        return {
            "model_state_dict": deepcopy(self.model.state_dict()),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
            "amp_scaler_state_dict": deepcopy(amp_state),
        }

    def _restore_training_state(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.use_amp and state.get("amp_scaler_state_dict") is not None:
            self.scaler.load_state_dict(state["amp_scaler_state_dict"])

    def _build_artifact_bundle(
        self,
        best_state: dict[str, Any],
        best_monitor: Optional[float],
        history: dict[str, list[Optional[float]]],
    ) -> dict[str, Any]:
        input_scaler_state = self.bundle.input_scaler.to_state().__dict__
        date_scaler_state = None
        if self.bundle.date_scaler is not None:
            date_scaler_state = self.bundle.date_scaler.to_state().__dict__

        export_cfg = deepcopy(self.cfg.export)
        export_cfg.feature_mode = self.cfg.windows.features
        export_cfg.target_column = self.cfg.windows.target

        return {
            "bundle_version": 1,
            "created_at": timestamp_tag(),
            "raw_csv_path": os.path.join(self.cfg.root_path, self.cfg.data_path),
            "window_config": asdict(self.cfg.windows),
            "model_config": asdict(self.cfg.model),
            "feature_config": asdict(self.cfg.feature_pipeline),
            "export_config": asdict(export_cfg),
            "optimizer_config": asdict(self.cfg.optim),
            "model_state_dict": best_state["model_state_dict"],
            "optimizer_state_dict": best_state["optimizer_state_dict"],
            "amp_scaler_state_dict": best_state.get("amp_scaler_state_dict"),
            "input_scaler_state": input_scaler_state,
            "date_scaler_state": date_scaler_state,
            "model_columns": list(self.bundle.model_columns),
            "target_column": self.bundle.target_column,
            "initial_live_state": deepcopy(self.bundle.initial_live_state),
            "split_paths": deepcopy(self.bundle.split_paths),
            "split_meta": deepcopy(self.bundle.split_meta),
            "best_monitor": best_monitor,
            "fit_history": history,
        }

    def fit(self) -> Dict[str, Any]:
        history: Dict[str, List[Optional[float]]] = {
            "train_loss": [],
            "val_loss": [],
        }

        print("\n=== LSTD Offline Fit ===")
        print(f"Run ID        : {self.run_id}")
        print(f"Device        : {self.device}")
        print(f"RAW CSV       : {os.path.join(self.cfg.root_path, self.cfg.data_path)}")
        print(f"Enc channels  : {self.bundle.enc_in}")
        print(f"Train windows : {len(self.bundle.train_dataset)}")
        print(f"Val windows   : {len(self.bundle.val_dataset) if self.bundle.val_dataset is not None else 0}")
        print(f"Artifact path : {self.artifact_bundle_path}")
        print()

        best_monitor: Optional[float] = None
        best_state: Optional[dict[str, Any]] = None
        patience_counter = 0

        epoch_bar = tqdm(
            range(self.cfg.optim.train_epochs),
            desc="Epochs",
            total=self.cfg.optim.train_epochs,
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            t0 = time.time()
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate_static()
            dt = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            monitor = train_loss if val_loss is None else val_loss
            improved = (best_monitor is None) or (monitor < best_monitor)

            if improved:
                best_monitor = monitor
                best_state = self._capture_training_state()
                torch.save(best_state["model_state_dict"], self.best_ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if val_loss is None:
                epoch_msg = f"train={train_loss:.6f} | val=SKIPPED | {dt:.1f}s"
            else:
                epoch_msg = f"train={train_loss:.6f} | val={val_loss:.6f} | {dt:.1f}s"

            print(f"Epoch {epoch + 1}/{self.cfg.optim.train_epochs} | {epoch_msg}")
            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=("NA" if val_loss is None else f"{val_loss:.4f}"),
                best=("NA" if best_monitor is None else f"{best_monitor:.4f}"),
                es=f"{patience_counter}/{self.cfg.patience}",
            )

            if patience_counter >= self.cfg.patience:
                print("Early stopping triggered.")
                break

        if best_state is None:
            best_state = self._capture_training_state()
            best_monitor = history["train_loss"][-1] if history["train_loss"] else None
            torch.save(best_state["model_state_dict"], self.best_ckpt_path)

        self._restore_training_state(best_state)

        artifact_bundle = self._build_artifact_bundle(best_state, best_monitor, history)
        torch.save(artifact_bundle, self.artifact_bundle_path)

        summary = {
            "run_id": self.run_id,
            "best_checkpoint": self.best_ckpt_path,
            "artifact_bundle_path": self.artifact_bundle_path,
            "best_monitor": best_monitor,
            "device": str(self.device),
            "raw_csv_path": os.path.join(self.cfg.root_path, self.cfg.data_path),
            "enc_in": self.bundle.enc_in,
            "target_index": self.bundle.target_index,
            "history": history,
            "window_config": asdict(self.cfg.windows),
            "model_config": asdict(self.cfg.model),
            "feature_config": asdict(self.cfg.feature_pipeline),
            "split_paths": self.bundle.split_paths,
            "split_meta": self.bundle.split_meta,
        }

        save_json(os.path.join(self.out_dir, "fit_summary.json"), summary)
        return summary
