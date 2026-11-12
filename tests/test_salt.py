import numpy as np
import pytest
import torch
from torch.nn import functional as F

from salt.losses import directional_loss, geometry_loss, image_loss, salt_loss
from salt.metrics import pixel_metrics, select_threshold
from salt.model import SALT, evidence, state_basis


def test_parameter_count_and_subspace():
    model = SALT().eval()
    assert sum(p.numel() for p in model.parameters()) == 67778
    features = torch.randn(1, 16, 1024)
    result = model(features)
    basis = result["basis"]
    torch.testing.assert_close(basis.T @ basis, torch.eye(2), atol=1e-6, rtol=1e-6)
    update = result["update"]
    torch.testing.assert_close(update, (update @ basis) @ basis.T, atol=1e-6, rtol=1e-5)
    expected = F.interpolate(result["patch_logits"].view(1, 1, 4, 4), size=(518, 518),
                             mode="bilinear", align_corners=False)[:, 0]
    torch.testing.assert_close(result["logits"], expected, rtol=0, atol=0)
    assert result["logits"].abs().max() <= 2


def test_collapsed_states_rejected():
    with pytest.raises(RuntimeError, match="collapsed"):
        state_basis(torch.ones(8), -torch.ones(8))


def test_soft_patch_occupancy_and_evidence():
    delta = torch.tensor([[0.2, -0.1, 0.0, 0.1]])
    mask = torch.tensor([[[[1.0, 0.5], [0.0, 1.0]]]])
    positive = (torch.tensor([1.0, 0.5, 0.0, 1.0]) * F.relu(0.1 - delta[0])).sum() / 2.5
    negative = (torch.tensor([0.0, 0.5, 1.0, 0.0]) * F.relu(0.1 + delta[0])).sum() / 1.5
    torch.testing.assert_close(directional_loss(delta, mask, 0.1), 0.5 * (positive + negative))
    features, positive_state, negative_state = torch.randn(1, 4, 8), torch.randn(8), torch.randn(8)
    expected = F.cosine_similarity(features, positive_state, dim=-1) - F.cosine_similarity(features, negative_state, dim=-1)
    torch.testing.assert_close(evidence(features, positive_state, negative_state), expected)


def test_pooling_before_sigmoid_and_geometry():
    logits = torch.tensor([[[2.0, -2.0], [0.5, 1.0]]])
    expected = F.binary_cross_entropy_with_logits(torch.tensor([1.5]), torch.ones(1))
    torch.testing.assert_close(image_loss(logits, torch.ones(1), 0.5), expected)
    features = torch.randn(2, 16, 8)
    assert float(geometry_loss(features, features)) == 0
    adapted = (features + torch.randn_like(features) * 0.1).requires_grad_()
    value = geometry_loss(features, adapted, seed=112)
    value.backward()
    assert adapted.grad.isfinite().all()
    assert adapted.grad.abs().sum() > 0


def test_complete_loss_backward():
    model = SALT(dim=16, hidden=4, resolution=8)
    output = model(torch.randn(2, 16, 16))
    config = {"topk_ratio": 0.0025, "geometry_pairs": 64, "direction_margin": 0.1,
              "lambda_dir": 0.5, "lambda_geo": 0.1}
    mask = torch.randint(2, (2, 1, 8, 8)).float()
    losses = salt_loss(output, model, mask, torch.ones(2), config, 1)
    losses["total"].backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_source_threshold_is_applied_unchanged():
    mask = np.array([0, 0, 1, 1], dtype=np.uint8)
    source = np.array([-0.8, 0.1, 0.2, 0.6], dtype=np.float32)
    threshold = select_threshold(mask, source)
    assert threshold == float(source[2])
    target = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    metrics = pixel_metrics(mask, target, threshold)
    assert metrics["strict_f1"] == pytest.approx(100 * 4 / 6)
    assert metrics["strict_iou"] == 50.0
    assert metrics["source_threshold"] == threshold
