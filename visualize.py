#!/usr/bin/env python3
"""Export RGB, ground truth, heatmap, and a source-thresholded prediction."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from salt.backbone import FrozenDINO
from salt.data import StructuralImages
from salt.utils import add_runtime_arguments, load_head, verify_weights, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0, help="Zero-based index in the fixed target test manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    verify_weights(args.weights)
    head, checkpoint = load_head(args.checkpoint, device)
    config = checkpoint["config"]
    backbone = FrozenDINO(args.dinov3_repo, args.weights).to(device).eval()
    dataset = StructuralImages(args.data_root, config["target"], "test", config["resolution"])
    batch = dataset[args.index]
    with torch.no_grad():
        score = head(backbone(batch["image"].unsqueeze(0).to(device)))["logits"][0].cpu().numpy()
    if config["sigma"] > 0:
        score = gaussian_filter(score, config["sigma"])
    mask = batch["mask"][0].numpy()
    prediction = score >= checkpoint["source_threshold"]
    with Image.open(dataset.root / batch["filename"]) as image:
        rgb = image.convert("RGB").resize((config["resolution"],) * 2, Image.Resampling.BICUBIC)
    args.output.mkdir(parents=True, exist_ok=True)
    rgb.save(args.output / "rgb.png")
    Image.fromarray(mask * 255).save(args.output / "ground_truth.png")
    Image.fromarray(prediction.astype(np.uint8) * 255).save(args.output / "prediction.png")
    np.save(args.output / "raw_scores.npy", score)
    # Normalization is for display only; the mask uses the raw score threshold.
    low, high = float(score.min()), float(score.max())
    display = (score - low) / max(high - low, 1e-12)
    plt.imsave(args.output / "heatmap.png", display, cmap="turbo", vmin=0, vmax=1)
    figure, axes = plt.subplots(1, 4, figsize=(10, 2.65))
    for axis, value, title in zip(axes, (rgb, mask, display, prediction),
                                  ("Image", "Ground truth", "SALT evidence", "Source threshold")):
        axis.imshow(value, cmap="turbo" if title == "SALT evidence" else "gray")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.tight_layout(pad=0.2)
    figure.savefig(args.output / "comparison.png", dpi=300, bbox_inches="tight")
    figure.savefig(args.output / "comparison.pdf", bbox_inches="tight")
    plt.close(figure)
    write_json(args.output / "sample.json", {
        "filename": batch["filename"], "index": args.index, "seed": checkpoint["seed"],
        "source_threshold": checkpoint["source_threshold"], "raw_min": low, "raw_max": high,
        "display_normalization_only": True,
    })
    print(args.output)


if __name__ == "__main__":
    main()
