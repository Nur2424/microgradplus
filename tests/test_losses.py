import math

import pytest
import torch

from microgradplus.engine import Value
from microgradplus.losses import bce_loss, hinge_loss, mse_loss


def test_mse_loss_basic():
    ypred = [Value(1.0), Value(2.0)]
    ytrue = [1.5, 1.5]
    loss = mse_loss(ypred, ytrue)
    # ((1-1.5)^2 + (2-1.5)^2) / 2 = (0.25 + 0.25)/2 = 0.25
    assert loss.data == pytest.approx(0.25)


def test_mse_loss_gradient_vs_pytorch():
    preds = [0.3, -1.2, 2.0]
    targets = [0.0, -1.0, 1.5]

    vpreds = [Value(p) for p in preds]
    loss = mse_loss(vpreds, targets)
    loss.backward()

    tpreds = torch.tensor(preds, requires_grad=True, dtype=torch.float64)
    ttargets = torch.tensor(targets, dtype=torch.float64)
    tloss = torch.mean((tpreds - ttargets) ** 2)
    tloss.backward()

    assert loss.data == pytest.approx(tloss.item())
    for v, g in zip(vpreds, tpreds.grad.tolist()):
        assert v.grad == pytest.approx(g)


def test_bce_loss_matches_pytorch():
    # ypred must be probabilities (post-sigmoid)
    logits = [0.8, -0.5, 2.0]
    targets = [1.0, 0.0, 1.0]

    vlogits = [Value(x) for x in logits]
    vprobs = [v.sigmoid() for v in vlogits]
    loss = bce_loss(vprobs, targets)
    loss.backward()

    tlogits = torch.tensor(logits, requires_grad=True, dtype=torch.float64)
    ttargets = torch.tensor(targets, dtype=torch.float64)
    tloss = torch.nn.functional.binary_cross_entropy_with_logits(tlogits, ttargets)
    tloss.backward()

    assert loss.data == pytest.approx(tloss.item(), abs=1e-8)
    for v, g in zip(vlogits, tlogits.grad.tolist()):
        assert v.grad == pytest.approx(g, abs=1e-8)


def test_bce_loss_gradient_is_pred_minus_true_at_logit():
    """The canonical result d(BCE)/d(logit) = (p - y) / N"""
    logits = [1.0, -2.0]
    targets = [1.0, 0.0]
    n = len(logits)

    vlogits = [Value(x) for x in logits]
    vprobs = [v.sigmoid() for v in vlogits]
    loss = bce_loss(vprobs, targets)
    loss.backward()

    for vlogit, vprob, y in zip(vlogits, vprobs, targets):
        expected = (vprob.data - y) / n
        assert vlogit.grad == pytest.approx(expected, abs=1e-8)


def test_hinge_loss_zero_when_confidently_correct():
    ypred = [Value(2.0), Value(-2.0)]
    ytrue = [1.0, -1.0]
    loss = hinge_loss(ypred, ytrue, margin=1.0)
    assert loss.data == pytest.approx(0.0)


def test_hinge_loss_positive_when_margin_violated():
    ypred = [Value(0.2)]
    ytrue = [1.0]
    loss = hinge_loss(ypred, ytrue, margin=1.0)
    # max(0, 1 - 1*0.2) = 0.8
    assert loss.data == pytest.approx(0.8)
