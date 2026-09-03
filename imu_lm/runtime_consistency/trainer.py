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
from imu_lm.utils.metrics import compute_metrics

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

        # Acc-only ablation: truncate every [B, C, T] batch to the first
        # data.sensor_columns channels (the acc-only prefix, by this repo's
        # acc-then-gyro column convention) before it reaches the objective, so
        # the encoder is trained/probed on accelerometer signal only. A single
        # choke point here covers MAE pretraining and every downstream
        # probe/finetune objective without touching per-source loaders or
        # per-objective forward_loss code. Some loaders already honor
        # data.sensor_columns and hand back acc-only data directly (in which
        # case this is a no-op); others (the FLU wrist loaders, the MAE
        # multidata cache) always emit the full acc+gyro concat regardless of
        # config, so this is the only reliable place to cut it down. Slicing
        # to an explicit target width (not just "half of whatever arrives")
        # keeps this idempotent either way.
        self._acc_only = bool(cfg_get(cfg, ["data", "acc_only"], False))
        self._acc_only_n = len(cfg_get(cfg, ["data", "sensor_columns"], []) or []) if self._acc_only else None
        if self._acc_only and not self._acc_only_n:
            raise ValueError("data.acc_only=true requires a non-empty data.sensor_columns to know the target channel count")

        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

        self.metrics_path = os.path.join(run_dir, "logs", "metrics.txt")
        self.ckpt_latest = os.path.join(run_dir, "checkpoints", "latest.pt")

        self.log_every = int(cfg_get(cfg, ["logging", "log_every_steps"], 100))
        self.ckpt_every = int(cfg_get(cfg, ["logging", "ckpt_every_steps"], 1000))
        self.save_step_checkpoints = bool(cfg_get(cfg, ["logging", "save_step_checkpoints"], True))
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
        # Selection metric for best.pt / early stopping. Default "val_loss"
        # (minimise) — the repo-wide convention. Any pooled val metric name
        # (e.g. "macro_f1", maximised) is also accepted; used to mirror the
        # OpenWatch paper's "early stop on val macro-F1" recipe (§5f).
        self.es_monitor = str(es_cfg.get("monitor", "val_loss"))
        # Direction: default keeps the historical rule (val_loss -> minimise,
        # anything else -> maximise). `early_stopping.mode: min|max` overrides
        # it so a lower-is-better pooled val metric (e.g. mpjpe_mm for the
        # pose tasks) can drive best.pt / early stopping (IMU-Bench uses this).
        es_mode = es_cfg.get("mode")
        if es_mode is None:
            self.es_monitor_maximise = self.es_monitor != "val_loss"
        else:
            if str(es_mode).lower() not in ("min", "max"):
                raise ValueError(f"early_stopping.mode must be 'min' or 'max', got {es_mode!r}")
            self.es_monitor_maximise = str(es_mode).lower() == "max"
        self.best_val_loss = float("inf")
        self.best_val_mpjpe = float("inf")
        self.epochs_without_improvement = 0
        self.ckpt_best = os.path.join(run_dir, "checkpoints", "best.pt")
        self.ckpt_best_mpjpe = os.path.join(run_dir, "checkpoints", "best_mpjpe.pt")
        # trainer.extra_best: {macro_f1: max, ...} -> also keep checkpoints/best_<metric>.pt
        _eb = (cfg_get(cfg, ["trainer", "extra_best"], {}) or {})
        self.extra_best_metrics = {str(k): (str(v).lower() == "max") for k, v in _eb.items()}
        self._extra_best_val = {}

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
        # Restore early stopping state + epoch from checkpoint if resuming.
        # Prefer latest.pt (freshest bookkeeping) but fall back to best.pt if
        # latest.pt is missing or corrupted (e.g. truncated by a disk-quota
        # write failure) -- both checkpoints carry the same bookkeeping keys.
        if start_step > 0:
            for ckpt_path in (self.ckpt_latest, self.ckpt_best):
                if not os.path.exists(ckpt_path):
                    continue
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                except Exception as e:
                    print(f"[resume] failed to read {ckpt_path} ({e}), trying next checkpoint")
                    continue
                self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
                self.best_val_mpjpe = ckpt.get("best_val_mpjpe", float("inf"))
                self.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
                self._epoch = ckpt.get("epoch", 0)
                print(f"[resume] epoch={self._epoch} best_val_loss={self.best_val_loss:.6f} patience_count={self.epochs_without_improvement} (from {ckpt_path})")
                break
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
                batch = self._acc_only_slice(batch)
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
                    if self.save_step_checkpoints:
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
        # Secondary "best_<metric>.pt" checkpoints, independent of early stopping: lets one run be
        # read under two selection rules (e.g. min val_loss AND max val macro_f1) without re-training.
        for m, maximise in getattr(self, "extra_best_metrics", {}).items():
            if avg_logs is None or m not in avg_logs:
                continue
            v = float(avg_logs[m]) * (-1.0 if maximise else 1.0)
            if v < self._extra_best_val.get(m, float("inf")):
                self._extra_best_val[m] = v
                self._save_ckpt(model, optimizer, scheduler, step, path=os.path.join(self.run_dir, "checkpoints", f"best_{m}.pt"))
                print(f"[train] new best val {m}={float(avg_logs[m]):.6f} -> best_{m}.pt")
        if self.early_stopping_enabled and val_loss is not None:
            # Score to compare: val_loss (min) or a pooled val metric (max),
            # stored negated so the "lower is better" bookkeeping is shared.
            if self.es_monitor == "val_loss":
                score = val_loss
            else:
                if avg_logs is None or self.es_monitor not in avg_logs:
                    raise KeyError(f"early_stopping.monitor={self.es_monitor!r} not in val logs "
                                   f"(have {sorted(avg_logs or {})})")
                score = float(avg_logs[self.es_monitor])
                if self.es_monitor_maximise:
                    score = -score
            if score < self.best_val_loss:
                self.best_val_loss = score
                self.epochs_without_improvement = 0
                self._save_ckpt(model, optimizer, scheduler, step, path=self.ckpt_best)
                if self.es_monitor != "val_loss":
                    print(f"[train] new best val {self.es_monitor}={(-score if self.es_monitor_maximise else score):.6f} (val_loss={val_loss:.6f})")
                else:
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
        # Classification objectives may attach raw per-batch predictions under
        # "_y_true"/"_y_pred" (list-valued, underscore-prefixed so the scalar
        # accumulation path above skips them). Macro-averaged metrics
        # (macro_f1/precision/recall, balanced accuracy) are NOT valid to
        # average per-batch and then mean across batches -- with many classes
        # and modest batch sizes, each batch's per-class support is small and
        # noisy, and F1 in particular is harshly punished by a single
        # zero-support class in one batch even if the model handles that
        # class fine over the full validation set. Pooling every prediction
        # across the whole eval pass and computing these metrics once, here,
        # is the statistically correct approach.
        pooled_y_true: list = []
        pooled_y_pred: list = []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = self._to_device(batch)
                batch = self._acc_only_slice(batch)
                with autocast("cuda", enabled=self.use_amp):
                    loss, logs = objective_step(batch, model, self.cfg)
                total_loss += float(loss.detach().item())
                count += 1
                if "_y_true" in logs and "_y_pred" in logs:
                    pooled_y_true.extend(logs["_y_true"])
                    pooled_y_pred.extend(logs["_y_pred"])
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
        if pooled_y_true:
            pooled = compute_metrics(pooled_y_true, pooled_y_pred)
            for k in ("bal_acc", "macro_f1", "macro_precision", "macro_recall"):
                if k in avg_logs and k in pooled:
                    avg_logs[k] = pooled[k]
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
            if k in ("loss", "epoch") or k.startswith("_"):
                continue  # "_"-prefixed: internal plumbing (e.g. pooled-metric inputs), not a metric itself
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

    def _acc_only_slice(self, batch):
        """Truncate batch[0] ([B, C, T]) to the first self._acc_only_n
        channels, and, if present, a matching-width bool channel_valid_mask
        ([B, C]) in batch[2]. No-op if the batch already arrives at the
        target width (loader already honored data.sensor_columns)."""
        if not self._acc_only or not isinstance(batch, tuple):
            return batch
        x = batch[0]
        if not (torch.is_tensor(x) and x.dim() == 3):
            return batch
        c = x.shape[1]
        n = self._acc_only_n
        if c <= n:
            return batch
        batch = list(batch)
        batch[0] = x[:, :n, :]
        if len(batch) > 2:
            m = batch[2]
            if torch.is_tensor(m) and m.dtype == torch.bool and m.dim() == 2 and m.shape[1] == c:
                batch[2] = m[:, :n]
        return tuple(batch)
