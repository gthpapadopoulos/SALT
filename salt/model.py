"""The SALA adapter and cosine-difference structural evidence."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def evidence(features: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    return (
        F.normalize(features, dim=-1)
        * (F.normalize(positive, dim=-1) - F.normalize(negative, dim=-1)).view(1, 1, -1)
    ).sum(-1)


def state_basis(negative: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    # Retain the original strided column layout and reduction order.
    raw = torch.stack((negative, positive), dim=1)
    normal = raw[:, 0].float() / (raw[:, 0].float().norm() + 1e-8)
    residual = raw[:, 1].float() - (raw[:, 1].float() * normal).sum() * normal
    norm = residual.norm()
    if float(norm.detach()) < 1e-4:
        raise RuntimeError(f"State basis collapsed: residual norm={float(norm.detach())}")
    return torch.stack((normal, residual / (norm + 1e-8)), dim=1)


class SALT(nn.Module):
    """Frozen-backbone features -> SALA correction -> raw dense evidence.

    q_N and q_A retain their original checkpoint names. They represent the
    negative and positive structural states, respectively.
    """

    def __init__(self, alpha=0.25, dim=1024, hidden=64, resolution=518):
        super().__init__()
        self.alpha = float(alpha)
        self.resolution = int(resolution)
        # Preserve the source implementation's shared state initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(111)
            self.q_A = nn.Parameter(torch.randn(dim) * (dim**-0.5) * 0.1)
            self.q_N = nn.Parameter(torch.randn(dim) * (dim**-0.5) * 0.1)
        self.coefficient = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 2)
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("Expected final-layer features with shape [B, N, d]")
        basis = state_basis(self.q_N, self.q_A)
        coefficient = self.coefficient(features)
        update = torch.matmul(coefficient, basis.T) * self.alpha
        adapted = features + update
        scores = evidence(adapted, self.q_A, self.q_N)
        side = math.isqrt(scores.shape[1])
        if side * side != scores.shape[1]:
            raise ValueError("Patch count must form a square grid")
        logits = F.interpolate(
            scores.view(scores.shape[0], 1, side, side),
            size=(self.resolution, self.resolution), mode="bilinear", align_corners=False,
        )[:, 0]
        return {
            "logits": logits, "patch_logits": scores, "z": features,
            "representation": adapted, "update": update, "basis": basis,
            "coefficient": coefficient,
        }
