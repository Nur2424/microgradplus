"""
loss functions over lists of `Value`s.

Each loss here corresponds to a maximum-likelihood assumption about the
noise/label distribution:

    Gaussian noise on a continuous target  => mean squared error
    Bernoulli label (binary classification) => binary cross-entropy
    Categorical label (multi-class)         => categorical cross-entropy
    Margin objective                        => hinge loss

New in this version:
    categorical_cross_entropy(logits, ytrue) takes raw unnormalized
    logits (a list of C Value objects per sample), applies softmax
    internally, and computes the negative log-likelihood of the correct
    class. The gradient w.r.t. each logit z_i collapses to (p_i - y_i)
"""

import math

from .engine import Value


def _as_value(v):
    return v if isinstance(v, Value) else Value(v)


def mse_loss(ypred, ytrue):
    """Mean squared error (1/N) * sum (yhat - y)^2.

    The maximum-likelihood loss under the assumption that targets are
    generated as `y = f(x) + noise`, with noise ~ N(0, sigma^2).
    Use for regression
    """
    n = len(ypred)
    total = sum((_as_value(yp) - yt) ** 2 for yp, yt in zip(ypred, ytrue))
    return total * (1.0 / n)


def _clamp_prob(p, eps=1e-12):
    """Clamp a Python float probability into (eps, 1 - eps) to avoid
    log(0). Operates on raw floats, not `Value`s, so it doesn't need a
    backward rule it only prevents a numerical edge case
    """
    return min(max(p, eps), 1 - eps)


def bce_loss(ypred, ytrue):
    """Binary cross-entropy: -(1/N) * sum [y*log(p) + (1-y)*log(1-p)].

    Expects ypred to already be probabilities in (0, 1) i.e. the
    model's last layer should use a sigmoid activation. The
    maximum-likelihood loss under a Bernoulli label model. Its gradient
    with respect to the *pre-sigmoid* logit is the canonical
    prediction target
    """
    n = len(ypred)
    total = Value(0.0)
    for yp, yt in zip(ypred, ytrue):
        p = _as_value(yp)
        if p.data <= 0.0 or p.data >= 1.0:
            p = Value(_clamp_prob(p.data), p._prev, p._op)
            p._backward = yp._backward if isinstance(yp, Value) else (lambda: None)
        term = yt * p.log() + (1 - yt) * (1 - p).log()
        total = total + term
    return total * (-1.0 / n)


def categorical_cross_entropy(logits_batch, ytrue):
    """Categorical cross-entropy loss for multi-class classification

    Args:
        logits_batch: list of N samples, each a list of C Value logits
            (raw unnormalized scores, one per class). Softmax is applied
            internally do NOT pass pre-softmaxed probabilities.
        ytrue: list of N integer class indices in [0, C-1]

    Returns:
        A single Value representing the mean loss over the batch:
            L = -(1/N) * sum_n log(p_{n, ytrue[n]})
        where p_n = softmax(logits_n)

    Why the gradient is (p_i - y_i):
        Let z_i be logit i, p_i = softmax(z_i), and y_i the one-hot
        target. The CE loss is L = -log(p_k) where k is the true class

        dL/dz_i = p_i - y_i  (i.e. p_i - 1 for i==k, p_i for i!=k)

        Derivation (sketch)
            p_k = exp(z_k) / Z,  Z = sum_j exp(z_j)
            log(p_k) = z_k - log(Z)
            d/dz_i [-log(p_k)] = d/dz_i [-z_k + log(Z)]
                                = -1{i==k} + exp(z_i)/Z
                                = p_i - y_i  (one-hot y_i).

        This is verified against PyTorch in tests/test_losses.py

    Numerical note:
        Value.softmax() subtracts max(logits) before exp for stability.
        The log(p_k) computation is -log(Z_shifted) + z_k_shifted,
        which flows correctly through the existing log/exp primitives
    """
    n = len(logits_batch)
    total = Value(0.0)

    for logits, k in zip(logits_batch, ytrue):
        probs = Value.softmax(logits)
        # Clamp the true-class probability away from 0 before log
        pk = probs[k]
        if pk.data < 1e-12:
            pk = Value(1e-12, pk._prev, pk._op)
        total = total + pk.log()

    # Mean negative log-likelihood
    return total * (-1.0 / n)


def hinge_loss(ypred, ytrue, margin=1.0):
    """Hinge loss (1/N) * sum max(0, margin - y * yhat).

    The perceptron/SVM-style loss for binary targets y in {-1, +1} and
    raw (unbounded) model outputs `yhat`. Zero once a point is classified
    correctly with confidence >= margin
    """
    n = len(ypred)
    total = Value(0.0)
    for yp, yt in zip(ypred, ytrue):
        margin_term = Value(margin) - yt * _as_value(yp)
        total = total + margin_term.relu()
    return total * (1.0 / n)
