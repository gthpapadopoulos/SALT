#!/usr/bin/env python3
"""Evaluate one frozen SALT checkpoint on the target test split."""

import argparse
import tempfile
from pathlib import Path

import torch

from salt.backbone import FrozenDINO
from salt.data import StructuralImages
from salt.engine import predict, profile_inference
from salt.metrics import all_metrics
from salt.utils import add_runtime_arguments, load_head, set_seed, verify_weights, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-predictions", action="store_true", help="Keep raw maps/masks; may require several GiB")
    parser.add_argument("--profile", action="store_true", help="Measure batch-one forward latency and peak allocated VRAM")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    verify_weights(args.weights)
    head, checkpoint = load_head(args.checkpoint, device)
    config = checkpoint["config"]
    set_seed(checkpoint["seed"])
    backbone = FrozenDINO(args.dinov3_repo, args.weights).to(device).eval()
    dataset = StructuralImages(args.data_root, config["target"], "test", config["resolution"])
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = None if args.save_predictions else tempfile.TemporaryDirectory(dir=args.output, prefix="predictions_")
    prediction_dir = args.output / "predictions" if scratch is None else Path(scratch.name)
    masks, maps = predict(head, backbone, dataset, config, device, prediction_dir)
    metrics = all_metrics(masks, maps, config["topk_ratio"], checkpoint["source_threshold"])
    del masks, maps
    if scratch is not None:
        scratch.cleanup()
    report = {"config": config, "seed": checkpoint["seed"], "selected_epoch": checkpoint["epoch"],
              "source_threshold": checkpoint["source_threshold"], "target_metrics": metrics,
              "threshold_origin": f"{config['source']}/val", "target_normalization": False,
              "target_threshold_search": False, "trainable_parameters": sum(p.numel() for p in head.parameters()),
              "checkpoint": str(args.checkpoint), "num_test_samples": len(dataset)}
    if args.profile:
        report["inference"] = profile_inference(head, backbone, dataset[0]["image"].unsqueeze(0), device)
    write_json(args.output / "result.json", report)
    if args.save_predictions:
        write_json(prediction_dir / "samples.json", [row["image_path"] for row in dataset.rows])
    print(f"{config['source']} -> {config['target']}, seed {checkpoint['seed']}: {metrics}")


if __name__ == "__main__":
    main()
