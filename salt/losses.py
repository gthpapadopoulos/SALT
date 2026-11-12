"""Source task supervision, Directional State Alignment, and geometry loss."""

import math

import torch
from torch.nn import functional as F

from .model import evidence


def focal_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    probability = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probability, 1.0 - probability)
    return ((1.0 - pt).pow(2.0) * bce).mean()


def dice_loss(logits, targets):
    probability = torch.sigmoid(logits).flatten(1)
    truth = targets.float().flatten(1)
    intersection = (probability * truth).sum(dim=1)
    denominator = probability.sum(dim=1) + truth.sum(dim=1)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def image_loss(logits, labels, ratio):
    flat = logits.flatten(1)
    k = max(1, int(flat.shape[1] * ratio))
    pooled = torch.topk(flat, k, dim=1).values.mean(dim=1)
    return F.binary_cross_entropy_with_logits(pooled, labels.float())


def directional_loss(delta, mask, margin, eps=1e-8):
    side = math.isqrt(delta.shape[1])
    occupancy = F.adaptive_avg_pool2d(mask.float(), (side, side))[:, 0].flatten(1)
    positive = (occupancy * F.relu(margin - delta)).sum() / (occupancy.sum() + eps)
    negative = ((1.0 - occupancy) * F.relu(margin + delta)).sum() / ((1.0 - occupancy).sum() + eps)
    return 0.5 * (positive + negative)


def geometry_loss(original, adapted, pair_count=4096, seed=111):
    generator = torch.Generator(device=original.device).manual_seed(seed)
    shape = (original.shape[0], pair_count)
    left = torch.randint(original.shape[1], shape, generator=generator, device=original.device)
    right = torch.randint(original.shape[1], shape, generator=generator, device=original.device)
    batch = torch.arange(original.shape[0], device=original.device)[:, None]
    before = F.cosine_similarity(original[batch, left], original[batch, right], dim=-1)
    after = F.cosine_similarity(adapted[batch, left], adapted[batch, right], dim=-1)
    return (after - before).abs().mean()


def salt_loss(output, head, mask, labels, config, epoch):
    logits = output["logits"]
    focal = focal_loss(logits, mask[:, 0])
    dice = dice_loss(logits, mask[:, 0])
    image = image_loss(logits, labels, config["topk_ratio"])
    # Keep the same normalization sequence as the original state loss.
    positive = F.normalize(F.normalize(head.q_A, dim=-1).expand(len(mask), -1), dim=-1)
    negative = F.normalize(F.normalize(head.q_N, dim=-1).expand(len(mask), -1), dim=-1)
    state = (F.cosine_similarity(positive, negative, dim=-1) + 0.5).clamp_min(0).mean()
    task = focal + dice + image + 0.1 * state
    baseline = evidence(output["z"], head.q_A, head.q_N)
    direction = directional_loss(output["patch_logits"] - baseline, mask, config["direction_margin"])
    geometry = geometry_loss(output["z"], output["representation"], config["geometry_pairs"], 111 + epoch)
    total = task + config["lambda_dir"] * direction + config["lambda_geo"] * geometry
    return {"total": total, "task": task, "focal": focal, "dice": dice,
            "image": image, "state": state, "direction": direction, "geometry": geometry}
