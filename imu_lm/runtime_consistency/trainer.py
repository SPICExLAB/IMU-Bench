"""Shared training loop (model-agnostic, objective_step contract)."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import random
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from imu_lm.utils.helpers import cfg_get

try:
    import wandb
except ImportError:
    wandb = None


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: Any, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = int(cfg_get(cfg, ["trainer", "seed"], 42))
        _set_seed(seed)

        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

        self.metrics_path = os.path.join(run_dir, "logs", "metrics.txt")
        self.ckpt_latest = os.path.join(run_dir, "checkpoints", "latest.pt")

        self.log_every = int(cfg_get(cfg, ["logging", "log_every_steps"], 100))
        self.ckpt_every = int(cfg_get(cfg, ["logging", "ckpt_every_steps"], 1000))
        self.max_steps = int(cfg_get(cfg, ["trainer", "max_steps"], 100000))
        self.use_amp = bool(cfg_get(cfg, ["trainer", "amp"], False))
        self.grad_clip_norm = float(cfg_get(cfg, ["trainer", "grad_clip_norm"], 0.0))
        self.grad_accum_steps = max(1, int(cfg_get(cfg, ["trainer", "grad_accum_steps"], 1)))
        self.val_every = int(cfg_get(cfg, ["trainer", "val_every_steps"], 0))
        # Max batches to consume per validation pass; 0 = no limit (use for finite datasets).
        # Set > 0 when val_loader is an infinite IterableDataset (e.g. var-len MAE, CLIP).
        self.val_max_batches = int(cfg_get(cfg, ["trainer", "val_max_batches"], 0))

        # Early stopping config
        es_cfg = cfg_get(cfg, ["trainer", "early_stopping"], {}) or {}
        self.early_stopping_enabled = bool(es_cfg.get("enabled", False))
        self.early_stopping_patience = int(es_cfg.get("patience", 5))
        self.best_val_loss = float("inf")
        self.best_val_mpjpe = float("inf")
        self.epochs_without_improvement = 0
        self.ckpt_best = os.path.join(run_dir, "checkpoints", "best.pt")
        self.ckpt_best_mpjpe = os.path.join(run_dir, "checkpoints", "best_mpjpe.pt")

    def fit(
        self,
        model: torch.nn.Module,
        objective_step: Callable[[Any, torch.nn.Module, Any], tuple],
        train_loader,
        val_loader=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        start_step: int = 0,
        extra_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ):
        if optimizer is None:
            raise ValueError("Trainer.fit requires an optimizer; got None")
        model.to(self.device)
        self._extra_modules = extra_modules or {}
        for name, mod in self._extra_modules.items():
            mod.to(self.device)
        self._all_params = list(model.parameters()) + [p for m in self._extra_modules.values() for p in m.parameters()]
        for state in optimizer.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        scaler = GradScaler("cuda", enabled=self.use_amp)

        step = int(start_step)
        self._epoch = 0
        # Restore early stopping state + epoch from checkpoint if resuming
        if start_step > 0:
            ckpt_path = self.ckpt_latest
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
                self.best_val_mpjpe = ckpt.get("best_val_mpjpe", float("inf"))
                self.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
                self._epoch = ckpt.get("epoch", 0)
                print(f"[resume] epoch={self._epoch} best_val_loss={self.best_val_loss:.6f} patience_count={self.epochs_without_improvement}")
        metrics_f = open(self.metrics_path, "a", buffering=1)

        stop = False
        # Compute total epochs so tqdm can show "X/Y ep" and an ETA.
        try:
            batches_per_epoch = len(train_loader)
        except TypeError:
            batches_per_epoch = None  # IterableDataset with no __len__
        remaining_steps = max(0, self.max_steps - step)
        if batches_per_epoch and batches_per_epoch > 0:
            import math
            total_epochs = self._epoch + math.ceil(remaining_steps / batches_per_epoch)
        else:
            total_epochs = None
        epoch_bar = (
            tqdm(total=total_epochs, initial=self._epoch, unit="ep",
                 dynamic_ncols=True, desc="training")
            if tqdm is not None else None
        )
        _last_loss = float("nan")

        while step < self.max_steps and not stop:
            self._epoch += 1
            if epoch_bar is not None:
                epoch_bar.set_description(f"epoch {self._epoch}")

            for batch in train_loader:
                if batch is None:
                    continue
                step += 1
                model.train()
                for mod in self._extra_modules.values():
                    mod.train()

                batch = self._to_device(batch)
                with autocast("cuda", enabled=self.use_amp):
                    loss, logs = objective_step(batch, model, self.cfg)
                    if self.grad_accum_steps > 1:
                        loss = loss / self.grad_accum_steps

                if self.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % self.grad_accum_steps == 0:
                    if self.use_amp:
                        scaler.unscale_(optimizer)
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self._all_params, self.grad_clip_norm)
                    if self.use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step()

                if step % self.log_every == 0:
                    lr = self._get_lr(optimizer)
                    logs["epoch"] = self._epoch
                    line = self._format_log(step, "train", lr, logs)
                    metrics_f.write(line + "\n")
                    print(line)
                    self._wandb_log("train", logs, step, lr=lr)
                    _last_loss = logs.get("loss", _last_loss)
                    if epoch_bar is not None:
                        epoch_bar.set_postfix(loss=f"{_last_loss:.4f}", step=step)

                if step % self.ckpt_every == 0:
                    step_path = os.path.join(
                        os.path.dirname(self.ckpt_latest),
                        f"step_{step:07d}.pt",
                    )
                    self._save_ckpt(model, optimizer, scheduler, step, path=step_path)
                    self._save_ckpt(model, optimizer, scheduler, step)

                # Periodic in-training validation
                if self.val_every > 0 and val_loader is not None and step % self.val_every == 0:
                    if self._run_val(val_loader, model, objective_step, optimizer, scheduler, step, metrics_f):
                        stop = True
                        break  # early stopping triggered

                if step >= self.max_steps:
                    break

            # Advance the epoch bar by one tick at the end of each epoch
            if epoch_bar is not None:
                epoch_bar.update(1)

            # Epoch-end validation (only if val_every_steps is not set)
            if not stop and val_loader is not None and self.val_every <= 0:
                if self._run_val(val_loader, model, objective_step, optimizer, scheduler, step, metrics_f):
                    stop = True

        metrics_f.close()
        if epoch_bar is not None:
            epoch_bar.close()
        # final checkpoint
        self._save_ckpt(model, optimizer, scheduler, step)
        reason = "early stopping" if stop else ("max_steps reached" if step >= self.max_steps else "data exhausted")
        print(f"[train] run complete: {reason} at step={step} epoch={self._epoch}")

    def _run_val(self, val_loader, model, objective_step, optimizer, scheduler, step, metrics_f):
        """Run validation, update scheduler/early-stopping. Returns True if early stopping triggered."""
        print(f"[train] eval at step {step}")
        val_loss, avg_logs = self._eval(val_loader, model, objective_step, step, metrics_f)
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau) and val_loss is not None:
            scheduler.step(val_loss)
        # Save best_mpjpe.pt whenever val mpjpe_mm improves (independent of early stopping)
        if avg_logs is not None and "mpjpe_mm" in avg_logs:
            mpjpe = avg_logs["mpjpe_mm"]
            if mpjpe < self.best_val_mpjpe:
                self.best_val_mpjpe = mpjpe
                self._save_ckpt(model, optimizer, scheduler, step, path=self.ckpt_best_mpjpe)
                print(f"[train] new best val mpjpe={mpjpe:.2f} mm")
        if self.early_stopping_enabled and val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_ckpt(model, optimizer, scheduler, step, path=self.ckpt_best)
                print(f"[train] new best val_loss={val_loss:.6f}")
            else:
                self.epochs_without_improvement += 1
                print(f"[train] no improvement for {self.epochs_without_improvement} evals")
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"[train] early stopping triggered at step {step}")
                    return True
        return False

    def _eval(self, val_loader, model, objective_step, step, metrics_f):
        model.eval()
        for mod in self._extra_modules.values():
            mod.eval()
        total_loss = 0.0
        count = 0
        all_logs_keys = set()
        accum_logs: Dict[str, float] = {}
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = self._to_device(batch)
                with autocast("cuda", enabled=self.use_amp):
                    loss, logs = objective_step(batch, model, self.cfg)
                total_loss += float(loss.detach().item())
                count += 1
                for k, v in logs.items():
                    if isinstance(v, (int, float)) and k != "loss":
                        accum_logs[k] = accum_logs.get(k, 0.0) + float(v)
                        all_logs_keys.add(k)
                if self.val_max_batches > 0 and count >= self.val_max_batches:
                    break
        if count == 0:
            return None, None
        avg_loss = total_loss / count
        avg_logs: Dict[str, float] = {"loss": avg_loss, "epoch": self._epoch}
        for k in sorted(all_logs_keys):
            avg_logs[k] = accum_logs[k] / count
        line = self._format_log(step, "val", self._get_lr(None), avg_logs)
        metrics_f.write(line + "\n")
        print(line)
        self._wandb_log("val", avg_logs, step)
        return avg_loss, avg_logs

    def _wandb_log(self, split: str, logs: Dict[str, float], step: int, lr: float = None):
        if wandb is None or wandb.run is None:
            return
        payload = {f"{split}/{k}": v for k, v in logs.items() if isinstance(v, (int, float, str))}
        if lr is not None:
            payload["lr"] = lr
        wandb.log(payload, step=step)

    def _get_lr(self, optimizer):
        if optimizer is None or len(optimizer.param_groups) == 0:
            return 0.0
        return optimizer.param_groups[0].get("lr", 0.0)

    def _format_log(self, step: int, split: str, lr: float, logs: Dict[str, float]):
        tokens = [f"epoch={logs.get('epoch', 0)}", f"step={step}", f"split={split}", f"loss={logs.get('loss', 0.0):.6f}", f"lr={lr:.6f}"]
        for k, v in logs.items():
            if k in ("loss", "epoch"):
                continue
            if isinstance(v, float):
                tokens.append(f"{k}={v:.6f}")
            else:
                tokens.append(f"{k}={v}")
        return " ".join(tokens)

    def _save_ckpt(self, model, optimizer, scheduler, step: int, path: Optional[str] = None):
        state = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "cfg": self.cfg,
            "best_val_loss": self.best_val_loss,
            "best_val_mpjpe": self.best_val_mpjpe,
            "epochs_without_improvement": self.epochs_without_improvement,
            "epoch": self._epoch,
        }
        for name, mod in self._extra_modules.items():
            state[name] = mod.state_dict()
        save_path = path or self.ckpt_latest
        torch.save(state, save_path)
        print(f"[train] saved checkpoint {save_path} (step={step})")

    def _to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return tuple(self._to_device(b) for b in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if torch.is_tensor(batch):
            return batch.to(self.device)
        return batch
