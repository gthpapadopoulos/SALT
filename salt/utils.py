"""Configuration and checkpoint utilities shared by the command-line tools."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .backbone import DINOV3_SHA256
from .model import SALT


def read_config(path):
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if {config["source"], config["target"]} != {"LEVIR", "WHU3"}:
        raise ValueError("Expected one of the two released transfer directions")
    if config["resolution"] != 518 or config["dim"] != 1024 or config["hidden"] != 64:
        raise ValueError("Use the released ViT-L/16 configuration")
    if not 0 < config["topk_ratio"] <= 1:
        raise ValueError("topk_ratio must be in (0, 1]")
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(path):
    digest = sha256(path)
    if digest != DINOV3_SHA256:
        raise ValueError(f"DINOv3 checkpoint hash mismatch: {digest}")
    return digest


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_head(config):
    return SALT(config["alpha"], config["dim"], config["hidden"], config["resolution"])


def load_head(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("config", "state", "source_threshold", "seed"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing {key}; use a released or train.py checkpoint")
    head = make_head(checkpoint["config"]).to(device)
    head.load_state_dict(checkpoint["state"], strict=True)
    return head.eval(), checkpoint


def add_runtime_arguments(parser):
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dinov3-repo", type=Path, default=Path(__file__).resolve().parents[1] / "third_party/dinov3")
    parser.add_argument("--device", default="cuda:0")
