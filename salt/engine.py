"""Prediction and resource measurements; all score maps remain raw logits."""

import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .data import make_loader


@torch.no_grad()
def predict(head, backbone, dataset, config, device, directory):
    """Store maps on disk to avoid holding an entire split in Python lists."""
    head.eval()
    backbone.eval()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shape = (len(dataset), config["resolution"], config["resolution"])
    maps = np.lib.format.open_memmap(directory / "maps.npy", mode="w+", dtype=np.float32, shape=shape)
    masks = np.lib.format.open_memmap(directory / "masks.npy", mode="w+", dtype=np.uint8, shape=shape)
    offset = 0
    for batch in make_loader(dataset, config):
        values = head(backbone(batch["image"].to(device, non_blocking=True)))["logits"].float().cpu().numpy()
        if config["sigma"] > 0:
            values = np.stack([gaussian_filter(value, config["sigma"]) for value in values])
        size = len(values)
        maps[offset:offset + size] = values
        masks[offset:offset + size] = batch["mask"][:, 0].numpy()
        offset += size
    maps.flush()
    masks.flush()
    return masks, maps


@torch.no_grad()
def profile_inference(head, backbone, image, device, warmup=10, repeats=50):
    head.eval()
    backbone.eval()
    image = image[:1].to(device)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        head(backbone(image))
    samples = []
    if cuda:
        torch.cuda.synchronize(device)
    for _ in range(repeats):
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            head(backbone(image))
            end.record()
            torch.cuda.synchronize(device)
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            head(backbone(image))
            samples.append(1000 * (time.perf_counter() - start))
    return {
        "batch_size": 1, "warmup": warmup, "repeats": repeats,
        "mean_ms_per_image": float(np.mean(samples)), "std_ms": float(np.std(samples, ddof=1)),
        "images_per_second": 1000 / float(np.mean(samples)),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "scope": "backbone and SALT forward pass; excludes loading, CPU smoothing, and metrics",
    }
