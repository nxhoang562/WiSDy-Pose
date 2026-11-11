"""Two-stage training of Fig. 4 and evaluation.

Stage 1: the autoencoder learns to map noisy CSI to clean CSI (MSE, Eq. 3) with RMSProp
and the learning-rate schedule of Eq. (5). Stage 2: its encoder is frozen as the Denoiser
and the Estimator is trained on the Denoiser's codes with Adam and an MSE loss on the
key-points. Both stages keep the weights with the lowest validation loss.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Callable

import numpy as np
import torch
from torch import nn

from .metrics import pose_metrics


def denoiser_lr_factor(epoch: int, constant_epochs: int = 45) -> float:
    """Eq. (5): zeta_0 for the first ``constant_epochs`` epochs, zeta_0 * e^-0.1 afterwards."""
    return 1.0 if epoch < constant_epochs else math.exp(-0.1)


def _run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            x, y = loss_fn.inputs(batch, device)
            loss = loss_fn(model(x), y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
    return total / max(count, 1)


class _ReconstructionLoss(nn.MSELoss):
    @staticmethod
    def inputs(batch, device):
        noisy, clean, _ = batch
        return noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)


class _KeypointLoss(nn.MSELoss):
    @staticmethod
    def inputs(batch, device):
        noisy, _, keypoints = batch
        return noisy.to(device, non_blocking=True), keypoints.to(device, non_blocking=True)


def _fit(model, optimizer, scheduler, loss_fn, train_loader, val_loader, epochs, device,
         log: Callable[[dict], None], prefix: str):
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = _run_epoch(model, val_loader, loss_fn, device)
        if scheduler is not None:
            scheduler.step()
        if val_loss < best:
            best, best_state = val_loss, copy.deepcopy(model.state_dict())
        log({"epoch": epoch, f"{prefix}/train_loss": train_loss, f"{prefix}/val_loss": val_loss,
             f"{prefix}/best_val_loss": best})
    model.load_state_dict(best_state)
    return best


def train_denoiser(autoencoder: nn.Module, train_loader, val_loader, cfg: dict, device,
                   log: Callable[[dict], None] = print) -> float:
    """Stage 1. Returns the best validation reconstruction MSE; the model keeps its weights."""
    d = cfg["denoiser"]
    autoencoder.to(device)
    opt = torch.optim.RMSprop(autoencoder.parameters(), lr=d["lr"], momentum=d["momentum"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: denoiser_lr_factor(e, d.get("lr_constant_epochs", 45)))
    return _fit(autoencoder, opt, sched, _ReconstructionLoss(), train_loader, val_loader,
                d["epochs"], device, log, "denoiser")


def train_estimator(model: nn.Module, train_loader, val_loader, cfg: dict, device,
                    log: Callable[[dict], None] = print) -> float:
    """Stage 2. Only the Estimator is optimised; the Denoiser stays frozen."""
    e = cfg["estimator"]
    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=e["lr"])
    return _fit(model, opt, None, _KeypointLoss(), train_loader, val_loader, e["epochs"],
                device, log, "estimator")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> dict:
    """Key-point loss, MPJPE / PA-MPJPE (mm), PCK@20-50 and inference time per sample."""
    model.to(device).eval()
    preds, gts, loss_sum, n, seconds = [], [], 0.0, 0, 0.0
    for noisy, _, keypoints in loader:
        x, y = noisy.to(device), keypoints.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        loss_sum += nn.functional.mse_loss(out, y, reduction="mean").item() * x.shape[0]
        n += x.shape[0]
        preds.append(out.cpu().numpy())
        gts.append(keypoints.numpy())
    metrics = pose_metrics(np.concatenate(preds), np.concatenate(gts))
    metrics.update(loss=loss_sum / n, ms_per_sample=1000.0 * seconds / n, n_samples=n)
    return metrics
