"""End-to-end experiment: data, stage 1, stage 2, test metrics, checkpoints."""
from __future__ import annotations

import json
import os
import random
from typing import Callable

import numpy as np
import torch
import yaml

from .data import build_datasets, make_loader
from .engine import evaluate, train_denoiser, train_estimator
from .models import PoseEstimator, build_autoencoder, build_estimator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict) -> PoseEstimator:
    """Untrained architecture of ``cfg`` (Denoiser + Estimator, or the Estimator alone
    when the config has no ``denoiser`` block)."""
    in_shape = (3, *cfg["data"].get("resize", (136, 32)))
    if cfg.get("denoiser"):
        ae = build_autoencoder(cfg)
        return PoseEstimator(build_estimator(cfg, ae.code_shape), ae.encoder)
    return PoseEstimator(build_estimator(cfg, in_shape))


def run_experiment(cfg: dict, data_root: str, out_dir: str | None = None, device: str | None = None,
                   log: Callable[[dict], None] = print) -> dict:
    """Train both stages on ``cfg`` and return validation and test metrics."""
    set_seed(cfg.get("seed", 0))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    workers = cfg["data"].get("num_workers", 4)
    train_set, val_set, test_set = build_datasets(cfg, data_root)
    results = {"n_train": len(train_set), "n_val": len(val_set), "n_test": len(test_set)}

    denoiser = None
    if cfg.get("denoiser"):
        bs = cfg["denoiser"]["batch_size"]
        ae = build_autoencoder(cfg)
        results["denoiser_val_mse"] = train_denoiser(
            ae, make_loader(train_set, bs, True, workers, cfg.get("seed", 0)),
            make_loader(val_set, bs, False, workers), cfg, dev, log)
        denoiser = ae.encoder
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(ae.state_dict(), os.path.join(out_dir, "autoencoder.pt"))

    in_shape = ae.code_shape if denoiser is not None else (3, *cfg["data"].get("resize", (136, 32)))
    model = PoseEstimator(build_estimator(cfg, in_shape), denoiser)
    bs = cfg["estimator"]["batch_size"]
    results["estimator_val_loss"] = train_estimator(
        model, make_loader(train_set, bs, True, workers, cfg.get("seed", 0)),
        make_loader(val_set, bs, False, workers), cfg, dev, log)
    results["val"] = evaluate(model, make_loader(val_set, bs, False, workers), dev)
    results["test"] = evaluate(model, make_loader(test_set, bs, False, workers), dev)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"config": cfg, "model_state_dict": model.state_dict()},
                   os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results
