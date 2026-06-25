import pytest

from microgradplus.engine import Value
from microgradplus.optim import SGD, Adam


def test_sgd_step_updates_data():
    p = Value(1.0)
    p.grad = 0.5
    opt = SGD([p], lr=0.1)
    opt.step()
    assert p.data == pytest.approx(1.0 - 0.1 * 0.5)


def test_sgd_zero_grad_resets():
    p = Value(1.0)
    p.grad = 5.0
    opt = SGD([p], lr=0.1)
    opt.zero_grad()
    assert p.grad == 0.0


def test_sgd_momentum_accumulates_velocity():
    p = Value(0.0)
    opt = SGD([p], lr=1.0, momentum=0.9)

    p.grad = 1.0
    opt.step()
    first_update = p.data  # -1.0

    p.grad = 1.0
    opt.step()
    second_step_delta = p.data - first_update
    # velocity grows: v1=1, v2 = 0.9*1 + 1 = 1.9, so second step is larger
    assert abs(second_step_delta) > abs(first_update)


def test_adam_reduces_simple_quadratic_loss():
    """Optimizing f(x) = (x - 3)^2 with Adam should converge toward x=3."""
    x = Value(0.0)
    opt = Adam([x], lr=0.1)

    for _ in range(200):
        opt.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        opt.step()

    assert x.data == pytest.approx(3.0, abs=1e-2)


def test_sgd_reduces_simple_quadratic_loss():
    x = Value(0.0)
    opt = SGD([x], lr=0.1)

    for _ in range(200):
        opt.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        opt.step()

    assert x.data == pytest.approx(3.0, abs=1e-2)
