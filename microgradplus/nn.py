"""
nn.py — Neuron, Layer, MLP, and MLPClassifier built on top of engine.Value.

New in this version:
    MLPClassifier — an MLP whose final layer outputs C raw logits
    (linear activation), designed to be used with categorical_cross_entropy.
    The number of output units equals the number of classes; pass logits
    directly to categorical_cross_entropy — do not apply softmax in the
    model forward pass (the loss function does it internally).
"""

import math
import random

from .engine import Value


class Module:
    """Base class providing the parameter-zeroing helper shared by all
    layers, mirroring the role of `nn.Module` / `optimizer.zero_grad()`.
    """

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


_ACTIVATIONS = {
    "tanh": lambda v: v.tanh(),
    "sigmoid": lambda v: v.sigmoid(),
    "relu": lambda v: v.relu(),
    "leaky_relu": lambda v: v.leaky_relu(),
    "linear": lambda v: v,
    None: lambda v: v,
}


def _init_weight(nin, scheme):
    if scheme == "uniform":
        return random.uniform(-1, 1)
    if scheme == "xavier":
        limit = math.sqrt(6.0 / nin)
        return random.uniform(-limit, limit)
    if scheme == "he":
        std = math.sqrt(2.0 / nin)
        return random.gauss(0.0, std)
    raise ValueError(f"unknown init scheme: {scheme!r}")


class Neuron(Module):
    """A single neuron: out = activation(w . x + b)."""

    def __init__(self, nin, activation="tanh", init="uniform"):
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; choose from "
                f"{sorted(a for a in _ACTIVATIONS if a)} or 'linear'"
            )
        self.w = [Value(_init_weight(nin, init)) for _ in range(nin)]
        self.b = Value(0.0 if init != "uniform" else random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return _ACTIVATIONS[self.activation](act)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation.capitalize() if self.activation else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """A layer of `nout` neurons, each seeing all `nin` inputs."""

    def __init__(self, nin, nout, activation="tanh", init="uniform"):
        self.neurons = [Neuron(nin, activation=activation, init=init) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A multi-layer perceptron.

    Args:
        nin: number of input features.
        nouts: list of layer widths, e.g. [16, 16, 1].
        activation: activation used by all hidden layers.
        out_activation: activation used by the final layer.
        init: weight initialization scheme.
    """

    def __init__(self, nin, nouts, activation="tanh", out_activation="tanh", init="uniform"):
        sz = [nin] + nouts
        layers = []
        for i in range(len(nouts)):
            is_last = i == len(nouts) - 1
            layers.append(
                Layer(
                    sz[i],
                    sz[i + 1],
                    activation=out_activation if is_last else activation,
                    init=init,
                )
            )
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


class MLPClassifier(Module):
    """Multi-class MLP that outputs a list of C raw logits.

    Identical to MLP except the output layer always uses 'linear'
    activation (no softmax applied here — categorical_cross_entropy
    applies softmax internally).

    Usage:
        model = MLPClassifier(nin=4, hidden=[16, 16], n_classes=3,
                              activation="relu", init="he")
        logits = model(x)           # list of 3 Value objects
        loss = categorical_cross_entropy([logits], [true_class_index])

    Why not apply softmax in the forward pass?
        Applying softmax in the model and log() in the loss creates a
        log(softmax(z)) expression. Numerically and analytically it is
        cleaner to compute log-softmax directly in the loss (which is
        what categorical_cross_entropy does via Value.softmax + log).
        This mirrors the PyTorch convention of CrossEntropyLoss expecting
        raw logits, not probabilities.
    """

    def __init__(self, nin, hidden, n_classes, activation="relu", init="he"):
        nouts = hidden + [n_classes]
        sz = [nin] + nouts
        layers = []
        for i in range(len(nouts)):
            is_last = i == len(nouts) - 1
            layers.append(
                Layer(
                    sz[i],
                    sz[i + 1],
                    activation="linear" if is_last else activation,
                    init=init,
                )
            )
        self.layers = layers
        self.n_classes = n_classes

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        # Always return a list, even for single-class edge case
        return x if isinstance(x, list) else [x]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def predict_class(self, x):
        """Return the predicted class index (argmax over logits)."""
        logits = self(x)
        return max(range(len(logits)), key=lambda i: logits[i].data)

    def __repr__(self):
        return f"MLPClassifier({self.n_classes} classes, [{', '.join(str(l) for l in self.layers)}])"
