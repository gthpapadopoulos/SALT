"""Load the pinned official DINOv3 ViT-L/16 implementation and weights."""

import importlib
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DINOV3_SHA256 = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"


class FrozenDINO(nn.Module):
    def __init__(self, repository, weights):
        super().__init__()
        repository, weights = Path(repository).resolve(), Path(weights).resolve()
        if not (repository / "dinov3/hub/backbones.py").is_file():
            raise FileNotFoundError(f"Missing official DINOv3 checkout: {repository}")
        if not weights.is_file():
            raise FileNotFoundError(f"Missing DINOv3 weights: {weights}")
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
        ).strip()
        if commit != DINOV3_COMMIT:
            raise ValueError(f"Expected DINOv3 revision {DINOV3_COMMIT}, got {commit}")
        sys.path.insert(0, str(repository))
        backbones = importlib.import_module("dinov3.hub.backbones")
        self.model = backbones.dinov3_vitl16(weights=str(weights))
        if len(self.model.blocks) != 24 or self.model.embed_dim != 1024:
            raise ValueError("Expected DINOv3 ViT-L/16")
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, image):
        return self.model.get_intermediate_layers(image, n=(23,), reshape=False, norm=True)[0].float()
