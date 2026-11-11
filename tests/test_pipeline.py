import copy
import json
import os

from wisdypose.pipeline import load_config, run_experiment


def _tiny(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["data"]["num_workers"] = 0
    for stage in ("denoiser", "estimator"):
        if cfg.get(stage):
            cfg[stage].update(epochs=2, batch_size=16)
    return cfg


def test_two_stage_training_end_to_end(fake_mmfi, tmp_path):
    cfg = _tiny(load_config("configs/protocol1_noise10.yaml"))
    res = run_experiment(cfg, fake_mmfi, str(tmp_path), device="cpu", log=lambda _: None)
    for key in ("mpjpe_mm", "pa_mpjpe_mm", "pck@20", "pck@50", "ms_per_sample"):
        assert key in res["test"]
    assert {"autoencoder.pt", "model.pt", "results.json"} <= set(os.listdir(tmp_path))
    assert json.load(open(tmp_path / "results.json"))["n_test"] == res["n_test"]


def test_estimator_only_on_clean_data(fake_mmfi):
    cfg = _tiny(load_config("configs/protocol1_original.yaml"))
    res = run_experiment(cfg, fake_mmfi, None, device="cpu", log=lambda _: None)
    assert "denoiser_val_mse" not in res and res["test"]["n_samples"] == res["n_test"]
