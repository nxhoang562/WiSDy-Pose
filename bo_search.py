"""Bayesian optimisation of the hyperparameters (Section IV-D, Algorithm 1).

Runs a Weights & Biases sweep (Gaussian-process surrogate, ``method: bayes``) over the
search space in ``configs/bo_search_space.yaml``. Every trial trains both stages and is
scored by the Estimator's validation loss; the test set is not used for the search.

    wandb login
    python bo_search.py --base configs/protocol1_noise10.yaml --data-root /path/to/MMFi --trials 30
"""
import argparse
import copy

import yaml

from wisdypose.pipeline import load_config, run_experiment


def trial_config(base: dict, p: dict) -> dict:
    """Merge one sampled point of the search space into the base config."""
    cfg = copy.deepcopy(base)
    if cfg.get("denoiser"):
        cfg["denoiser"].update(
            lr=p["ae_lr"], momentum=p["ae_momentum"], batch_size=p["ae_batch_size"], epochs=p["ae_epochs"],
            n_kernels=[p["ae_n_kernels_1"], p["ae_n_kernels_2"]],
            kernel_sizes=[p["ae_kernel_size_1"], p["ae_kernel_size_2"]],
            pool_sizes=[p["ae_maxpooling_1"], p["ae_maxpooling_2"]])
    n = p["n_stacks"]
    cfg["estimator"].update(
        lr=p["est_lr"], dropout=p["est_dropout"], temperature=p["est_temp"], batch_size=p["est_batch_size"],
        epochs=p["est_epochs"], n_basis_kernels=p["est_n_basis_kernels"], d_hidden=p["est_d_hidden"],
        n_kernels=[p[f"est_n_kernels_{i}"] for i in range(1, n + 1)],
        kernel_sizes=[p[f"est_kernel_size_{i}"] for i in range(1, n + 1)],
        pool_sizes=[p[f"est_maxpooling_{i}"] for i in range(1, n + 1)])
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="config giving the data setting (protocol, noise, split)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--space", default="configs/bo_search_space.yaml")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--project", default="wisdy-pose")
    args = ap.parse_args()

    import wandb

    base = load_config(args.base)
    with open(args.space) as f:
        sweep = yaml.safe_load(f)
    if not base.get("denoiser"):   # clean-data setting: search the Estimator only
        sweep["parameters"] = {k: v for k, v in sweep["parameters"].items() if not k.startswith("ae_")}

    def trial():
        with wandb.init() as run:
            res = run_experiment(trial_config(base, dict(run.config)), args.data_root, log=wandb.log)
            wandb.log({"estimator_val_loss": res["estimator_val_loss"],
                       **{f"val/{k}": v for k, v in res["val"].items()}})

    sweep_id = wandb.sweep(sweep, project=args.project)
    wandb.agent(sweep_id, function=trial, count=args.trials)


if __name__ == "__main__":
    main()
