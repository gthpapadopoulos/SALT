#!/usr/bin/env python3
"""Train SALT using source training and validation data only."""

import argparse
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from salt.backbone import DINOV3_COMMIT, FrozenDINO
from salt.data import StructuralImages, epoch_indices, make_loader
from salt.engine import predict
from salt.losses import salt_loss
from salt.metrics import all_metrics, pixel_metrics, select_threshold
from salt.utils import add_runtime_arguments, make_head, read_config, set_seed, verify_weights, write_json


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(111, 222, 333), default=111)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt at the last completed epoch")
    args = parser.parse_args()
    config = read_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "train_result.json").exists():
        raise FileExistsError("Training is complete in this output directory")
    if (args.output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Existing run found; use --resume or a fresh output directory")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    digest = verify_weights(args.weights)
    set_seed(args.seed)
    backbone = FrozenDINO(args.dinov3_repo, args.weights).to(device).eval()
    head = make_head(config).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    train_set = StructuralImages(args.data_root, config["source"], "train", config["resolution"], augmented=True)
    val_set = StructuralImages(args.data_root, config["source"], "val", config["resolution"])
    provenance = {"dinov3_commit": DINOV3_COMMIT, "backbone_sha256": digest,
                  "selection_split": f"{config['source']}/val", "target_used_for_selection": False}
    history, best, first_epoch, previous_seconds = [], (-float("inf"), -float("inf")), 1, 0.0
    if args.resume:
        # Resume files contain RNG states and are trusted local training outputs.
        last = torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)
        if last["config"] != config or last["seed"] != args.seed:
            raise ValueError("Resume configuration or seed differs from the saved run")
        head.load_state_dict(last["state"])
        optimizer.load_state_dict(last["optimizer"])
        history, best = last["history"], tuple(last["best"])
        first_epoch, previous_seconds = last["epoch"] + 1, last["training_seconds"]
        random.setstate(last["random_state"])
        np.random.set_state(last["numpy_state"])
        torch.set_rng_state(last["torch_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(last["cuda_state"], device)
    write_json(args.output / "config.json", {"config": config, "seed": args.seed, **provenance})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(first_epoch, config["epochs"] + 1):
        head.train()
        totals, count = {}, 0
        indices = epoch_indices(len(train_set.rows), epoch, args.seed)
        for batch in make_loader(train_set, config, indices):
            optimizer.zero_grad(set_to_none=True)
            features = backbone(batch["image"].to(device, non_blocking=True))
            output = head(features)
            mask = batch["mask"].to(device, non_blocking=True).float()
            labels = batch["label"].to(device, non_blocking=True).float()
            losses = salt_loss(output, head, mask, labels, config, epoch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            count += len(mask)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(mask)
        with tempfile.TemporaryDirectory(dir=args.output, prefix="validation_") as scratch:
            masks, maps = predict(head, backbone, val_set, config, device, scratch)
            metrics = pixel_metrics(masks, maps)
            del masks, maps
        key = (metrics["pixel_ap"], metrics["pixel_auroc"])
        checkpoint = {"state": head.state_dict(), "config": config, "seed": args.seed,
                      "epoch": epoch, "provenance": provenance}
        if key > best:
            best = key
            save_checkpoint(args.output / "best.pt", checkpoint)
        history.append({"epoch": epoch, **{k: v / count for k, v in totals.items()}, **metrics})
        write_json(args.output / "history.json", history)
        checkpoint.update(
            optimizer=optimizer.state_dict(), history=history, best=best,
            training_seconds=previous_seconds + time.perf_counter() - started,
            random_state=random.getstate(), numpy_state=np.random.get_state(),
            torch_state=torch.get_rng_state(),
            cuda_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        )
        save_checkpoint(args.output / "last.pt", checkpoint)
        print(f"Epoch {epoch:02d}: source P-AP={metrics['pixel_ap']:.4f}, P-AUROC={metrics['pixel_auroc']:.4f}", flush=True)
    selected = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(selected["state"])
    with tempfile.TemporaryDirectory(dir=args.output, prefix="calibration_") as scratch:
        masks, maps = predict(head, backbone, val_set, config, device, scratch)
        threshold = select_threshold(masks, maps)
        metrics = all_metrics(masks, maps, config["topk_ratio"], threshold)
        del masks, maps
    selected.update(source_threshold=threshold, source_metrics=metrics)
    save_checkpoint(args.output / "best.pt", selected)
    report = {"config": config, "seed": args.seed, "selected_epoch": selected["epoch"],
              "source_threshold": threshold, "source_metrics": metrics, "provenance": provenance,
              "trainable_parameters": sum(p.numel() for p in head.parameters()),
              "training_seconds": previous_seconds + time.perf_counter() - started,
              "training_peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    write_json(args.output / "train_result.json", report)
    print(f"Saved {args.output / 'best.pt'}; source threshold={threshold:.8f}", flush=True)


if __name__ == "__main__":
    main()
