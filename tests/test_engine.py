"""
Tests for the core autograd engine

Two complementary checks for every gradient claim:
  - Compare against PyTorch's autograd on the same expression
  - Compare against a finite-difference numerical derivative
"""

import math

import pytest
import torch

from microgradplus.engine import Value


def test_basic_forward():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    assert d.data == pytest.approx(4.0)


def test_add_mul_backward_vs_pytorch():
    a = Value(-2.0)
    b = Value(3.0)
    d = a * b
    e = a + b
    f = d * e
    f.backward()

    ta = torch.tensor(-2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    td = ta * tb
    te = ta + tb
    tf = td * te
    tf.backward()

    assert f.data == pytest.approx(tf.item())
    assert a.grad == pytest.approx(ta.grad.item())
    assert b.grad == pytest.approx(tb.grad.item())


def test_gradient_accumulation_when_value_used_twice():
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == pytest.approx(2.0)


def test_pow_and_div():
    a = Value(4.0)
    b = a ** 2
    c = b / a  # = a
    c.backward()
    assert c.data == pytest.approx(4.0)
    assert a.grad == pytest.approx(1.0)


@pytest.mark.parametrize("x", [-1.5, 0.0, 0.7, 2.3])
def test_tanh_vs_pytorch(x):
    v = Value(x)
    out = v.tanh()
    out.backward()

    t = torch.tensor(x, requires_grad=True, dtype=torch.float64)
    tout = torch.tanh(t)
    tout.backward()

    assert out.data == pytest.approx(tout.item(), abs=1e-10)
    assert v.grad == pytest.approx(t.grad.item(), abs=1e-10)


@pytest.mark.parametrize("x", [-2.0, 0.0, 1.3])
def test_sigmoid_vs_pytorch(x):
    v = Value(x)
    out = v.sigmoid()
    out.backward()

    t = torch.tensor(x, requires_grad=True, dtype=torch.float64)
    tout = torch.sigmoid(t)
    tout.backward()

    assert out.data == pytest.approx(tout.item(), abs=1e-10)
    assert v.grad == pytest.approx(t.grad.item(), abs=1e-10)


@pytest.mark.parametrize("x", [-1.0, 0.5, 2.0])
def test_relu_vs_pytorch(x):
    v = Value(x)
    out = v.relu()
    out.backward()

    t = torch.tensor(x, requires_grad=True, dtype=torch.float64)
    tout = torch.relu(t)
    tout.backward()

    assert out.data == pytest.approx(tout.item())
    assert v.grad == pytest.approx(t.grad.item())


def test_leaky_relu_negative_branch():
    v = Value(-2.0)
    out = v.leaky_relu(alpha=0.1)
    out.backward()
    assert out.data == pytest.approx(-0.2)
    assert v.grad == pytest.approx(0.1)


def test_tanh_decomposed_matches_atomic_tanh():
    """tanh built from exp/+/-// should give identical gradients to the
    atomic tanh implementation i.e. no operation is privileged."""
    x1, w1 = Value(2.0), Value(-3.0)

    n1 = x1 * w1
    o1 = n1.tanh()
    o1.backward()

    x2, w2 = Value(2.0), Value(-3.0)
    n2 = x2 * w2
    e = (2 * n2).exp()
    o2 = (e - 1) / (e + 1)
    o2.backward()

    assert o1.data == pytest.approx(o2.data)
    assert x1.grad == pytest.approx(x2.grad)
    assert w1.grad == pytest.approx(w2.grad)


def test_finite_difference_matches_backward():
    """Sanity check: d/dx of f(x) = (x^2 + 3) * tanh(x) at x=1.2,
    via backward() vs central finite differences"""

    def f(x):
        return (x ** 2 + 3) * math.tanh(x)

    x0 = 1.2
    h = 1e-6
    numerical = (f(x0 + h) - f(x0 - h)) / (2 * h)

    v = Value(x0)
    out = (v ** 2 + 3) * v.tanh()
    out.backward()

    assert v.grad == pytest.approx(numerical, abs=1e-5)


def test_mlp_neuron_gradient_vs_pytorch():
    """End-to-end check on a single tanh neuron, mirroring section 12 of
    the notebook"""
    x1v, x2v = 2.0, 0.0
    w1v, w2v, bv = -3.0, 1.0, 6.8813735870195432

    x1 = Value(x1v)
    x2 = Value(x2v)
    w1 = Value(w1v)
    w2 = Value(w2v)
    b = Value(bv)
    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()
    o.backward()

    tx1 = torch.tensor(x1v, requires_grad=True, dtype=torch.float64)
    tx2 = torch.tensor(x2v, requires_grad=True, dtype=torch.float64)
    tw1 = torch.tensor(w1v, requires_grad=True, dtype=torch.float64)
    tw2 = torch.tensor(w2v, requires_grad=True, dtype=torch.float64)
    tb = torch.tensor(bv, requires_grad=True, dtype=torch.float64)
    tn = tx1 * tw1 + tx2 * tw2 + tb
    to = torch.tanh(tn)
    to.backward()

    assert o.data == pytest.approx(to.item(), abs=1e-10)
    for v, t in [(x1, tx1), (x2, tx2), (w1, tw1), (w2, tw2), (b, tb)]:
        assert v.grad == pytest.approx(t.grad.item(), abs=1e-10)
