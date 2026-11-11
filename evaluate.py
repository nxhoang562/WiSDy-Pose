"""Evaluate a trained model (``model.pt`` written by train.py) on its test split.

    python evaluate.py --checkpoint runs/protocol1_noise10_random/model.pt --data-root /path/to/MMFi
"""
import argparse
import json

import torch

from wisdypose.data import build_datasets, make_loader
from wisdypose.engine import evaluate
from wisdypose.pipeline import build_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--subset", choices=["val", "test"], default="test")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    _, val_set, test_set = build_datasets(cfg, args.data_root)
    loader = make_loader(val_set if args.subset == "val" else test_set, cfg["estimator"]["batch_size"],
                         False, cfg["data"].get("num_workers", 4))
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(json.dumps(evaluate(model, loader, dev), indent=2))


if __name__ == "__main__":
    main()
