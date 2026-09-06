"""
test_softmax_cce.py correctness tests for softmax + categorical cross-entropy

Every gradient is verified two ways:
 - Against PyTorch's autograd on identical expressions
 - Against finite-difference numerical derivatives 

Tests:
    test_softmax_probabilities_sum_to_one
    test_softmax_gradients_match_pytorch
    test_softmax_gradient_is_p_minus_y           <= the key algebraic identity
    test_cce_loss_matches_pytorch
    test_cce_gradient_matches_pytorch
    test_cce_gradient_is_p_minus_y_full_batch
    test_finite_difference_softmax
    test_finite_difference_cce
    test_mlpclassifier_forward_shape
    test_mlpclassifier_training_reduces_loss
"""

import math
import random
import pytest

# ----- micrograd+ ------
from microgradplus.engine import Value
from microgradplus.losses import categorical_cross_entropy
from microgradplus.nn import MLPClassifier
from microgradplus.optim import Adam
from microgradplus.training import fit

# ----- PyTorch for reference -----
import torch
import torch.nn.functional as F


# ----- Helpers -----

def _value_logits(data):
    """Create a list of Value logits from a list of floats"""
    return [Value(d) for d in data]


def _torch_logits(data):
    """Create a torch tensor of logits with grad tracking"""
    return torch.tensor(data, dtype=torch.float64, requires_grad=True)


def finite_diff_grad(fn, x_list, idx, eps=1e-5):
    """Numerically estimate df/dx[idx] using central differences"""
    x_plus = [v + (eps if i == idx else 0.0) for i, v in enumerate(x_list)]
    x_minus = [v - (eps if i == idx else 0.0) for i, v in enumerate(x_list)]
    return (fn(x_plus) - fn(x_minus)) / (2 * eps)


# ----- 1 Softmax probabilities -----

def test_softmax_probabilities_sum_to_one():
    logits = _value_logits([2.0, 1.0, 0.1])
    probs = Value.softmax(logits)
    total = sum(p.data for p in probs)
    assert abs(total - 1.0) < 1e-9, f"probs sum to {total}, expected 1.0"


def test_softmax_values_match_pytorch():
    data = [2.0, 1.0, 0.1]
    probs = Value.softmax(_value_logits(data))
    t_probs = F.softmax(torch.tensor(data, dtype=torch.float64), dim=0)
    for p, tp in zip(probs, t_probs):
        assert abs(p.data - tp.item()) < 1e-9



# ----- 2 Softmax gradients via backward -----

def test_softmax_gradients_match_pytorch():
    """d(sum(probs * weights)) / d(logit_i) matches PyTorch"""
    data = [2.0, 1.0, 0.1]
    weights = [3.0, -1.0, 0.5]   # arbitrary output weights

    # micrograd+
    logits = _value_logits(data)
    probs = Value.softmax(logits)
    out = sum(p * w for p, w in zip(probs, weights))
    out.backward()
    mg_grads = [l.grad for l in logits]

    # PyTorch
    t_logits = _torch_logits(data)
    t_probs = F.softmax(t_logits, dim=0)
    t_out = (t_probs * torch.tensor(weights, dtype=torch.float64)).sum()
    t_out.backward()
    pt_grads = t_logits.grad.tolist()

    for mg, pt in zip(mg_grads, pt_grads):
        assert abs(mg - pt) < 1e-7, f"grad mismatch: mg={mg:.6f} pt={pt:.6f}"


# ----- 3 The key identity: dL/dz_i = p_i - y_i -----

def test_softmax_gradient_is_p_minus_y():
    """For CE loss on a single sample, dL/dz_i == p_i - y_i exactly"""
    data = [2.0, 1.0, 0.1]
    true_class = 0  # y = [1, 0, 0]

    logits = _value_logits(data)
    loss = categorical_cross_entropy([logits], [true_class])
    loss.backward()

    probs = F.softmax(torch.tensor(data, dtype=torch.float64), dim=0).tolist()
    y_onehot = [1.0 if i == true_class else 0.0 for i in range(len(data))]

    for i, (l, p, y) in enumerate(zip(logits, probs, y_onehot)):
        expected = p - y
        assert abs(l.grad - expected) < 1e-7, (
            f"class {i}: grad={l.grad:.6f}, p_i - y_i={expected:.6f}"
        )



# ----- 4 CCE loss value matches PyTorch -----


def test_cce_loss_matches_pytorch():
    """Loss value == PyTorch CrossEntropyLoss on the same inputs"""
    batch = [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3], [-1.0, 0.0, 3.0]]
    targets = [0, 1, 2]

    # micrograd+
    logits_batch = [_value_logits(row) for row in batch]
    loss_mg = categorical_cross_entropy(logits_batch, targets)

    # PyTorch
    t_logits = torch.tensor(batch, dtype=torch.float64)
    t_targets = torch.tensor(targets, dtype=torch.long)
    loss_pt = torch.nn.CrossEntropyLoss()(t_logits, t_targets)

    assert abs(loss_mg.data - loss_pt.item()) < 1e-7, (
        f"loss mismatch: mg={loss_mg.data:.6f} pt={loss_pt.item():.6f}"
    )


def test_cce_gradient_matches_pytorch():
    """CCE gradients w.r.t. every logit match PyTorch exactly"""
    batch = [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]
    targets = [0, 1]

    # micrograd+
    logits_batch = [_value_logits(row) for row in batch]
    loss_mg = categorical_cross_entropy(logits_batch, targets)
    loss_mg.backward()
    mg_grads = [[l.grad for l in row] for row in logits_batch]

    # PyTorch
    t_logits = torch.tensor(batch, dtype=torch.float64, requires_grad=True)
    t_targets = torch.tensor(targets, dtype=torch.long)
    loss_pt = torch.nn.CrossEntropyLoss()(t_logits, t_targets)
    loss_pt.backward()
    pt_grads = t_logits.grad.tolist()

    for i, (mg_row, pt_row) in enumerate(zip(mg_grads, pt_grads)):
        for j, (mg, pt) in enumerate(zip(mg_row, pt_row)):
            assert abs(mg - pt) < 1e-7, (
                f"sample {i} class {j}: mg={mg:.6f} pt={pt:.6f}"
            )


# ----- 5 p_i - y_i identity over full batch (mean CCE) ------


def test_cce_gradient_is_p_minus_y_full_batch():
    """dL/dz_{n,i} == (p_{n,i} - y_{n,i}) / N for mean CCE loss"""
    batch = [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3], [-1.0, 0.0, 3.0]]
    targets = [0, 1, 2]
    N = len(batch)

    logits_batch = [_value_logits(row) for row in batch]
    loss = categorical_cross_entropy(logits_batch, targets)
    loss.backward()

    for n, (row_data, k) in enumerate(zip(batch, targets)):
        t_probs = F.softmax(torch.tensor(row_data, dtype=torch.float64), dim=0).tolist()
        for i, (logit, p) in enumerate(zip(logits_batch[n], t_probs)):
            y_i = 1.0 if i == k else 0.0
            expected = (p - y_i) / N
            assert abs(logit.grad - expected) < 1e-7, (
                f"sample {n} class {i}: grad={logit.grad:.7f} expected={expected:.7f}"
            )


# ------ 6 Finite-difference checks -----

def test_finite_difference_softmax():
    """Finite-difference verification of softmax gradients"""
    data = [1.5, -0.5, 2.0]
    weights = [1.0, -2.0, 0.5]

    def forward(z):
        logits = _value_logits(z)
        probs = Value.softmax(logits)
        return sum(p.data * w for p, w in zip(probs, weights))

    logits = _value_logits(data)
    probs = Value.softmax(logits)
    out = sum(p * w for p, w in zip(probs, weights))
    out.backward()

    for i in range(len(data)):
        fd = finite_diff_grad(forward, data, i)
        assert abs(logits[i].grad - fd) < 1e-5, (
            f"logit {i}: autograd={logits[i].grad:.6f} fd={fd:.6f}"
        )


def test_finite_difference_cce():
    """Finite-difference verification of CCE gradients"""
    data = [2.0, 0.5, -1.0]
    k = 1

    def forward(z):
        logits = _value_logits(z)
        loss = categorical_cross_entropy([logits], [k])
        return loss.data

    logits = _value_logits(data)
    loss = categorical_cross_entropy([logits], [k])
    loss.backward()

    for i in range(len(data)):
        fd = finite_diff_grad(forward, data, i)
        assert abs(logits[i].grad - fd) < 1e-5, (
            f"logit {i}: autograd={logits[i].grad:.6f} fd={fd:.6f}"
        )


# ------ 7 MLPClassifier integration tests ------


def test_mlpclassifier_forward_shape():
    """MLPClassifier returns a list of n_classes Values per sample """
    random.seed(0)
    model = MLPClassifier(nin=4, hidden=[8], n_classes=3)
    x = [0.5, -0.3, 1.2, 0.0]
    logits = model(x)
    assert len(logits) == 3
    assert all(isinstance(l, Value) for l in logits)


def test_mlpclassifier_training_reduces_loss():
    """Training for 50 epochs on a 3-class toy dataset reduces the loss """
    random.seed(42)
    # XOR-like 3-class toy 3 groups of 4 points
    X = [
        [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.0],  # class 0
        [-1.0, 1.0], [-1.1, 0.9], [-0.9, 1.1], [-1.0, 1.0],  # class 1
        [0.0, -1.0], [0.1, -1.1], [-0.1, -0.9], [0.0, -1.0],  # class 2
    ]
    y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    model = MLPClassifier(nin=2, hidden=[8, 8], n_classes=3,
                          activation="relu", init="he")
    opt = Adam(model.parameters(), lr=0.05)

    history = fit(model, opt, categorical_cross_entropy, X, y,
                  epochs=80, verbose=False, multiclass=True)

    assert history[-1] < history[0], (
        f"loss did not decrease: start={history[0]:.4f} end={history[-1]:.4f}"
    )
    # Should get to at least below 1.0 on this trivially separable dataset
    assert history[-1] < 1.0, f"final loss too high: {history[-1]:.4f}"
