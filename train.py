"""Train the Denoiser and the Estimator, then report validation and test metrics.

    python train.py --config configs/protocol1_noise10.yaml --data-root /path/to/MMFi
    python train.py --config configs/protocol1_original.yaml --data-root /path/to/MMFi --split cross_subject
"""
import argparse
import json

from wisdypose.pipeline import load_config, run_experiment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML experiment config (see configs/)")
    ap.add_argument("--data-root", required=True, help="MM-Fi root holding E01 ... E04")
    ap.add_argument("--out", default=None, help="output directory (default: runs/<config name>)")
    ap.add_argument("--split", choices=["random", "cross_subject", "cross_environment"], default=None,
                    help="override data.split (S1 / S2 / S3)")
    ap.add_argument("--device", default=None, help="cuda, cuda:1, cpu, ... (default: cuda if available)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.split:
        cfg["data"]["split"] = args.split
    name = args.config.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    out = args.out or f"runs/{name}_{cfg['data']['split']}"
    results = run_experiment(cfg, args.data_root, out, args.device)
    print(json.dumps(results["test"], indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
