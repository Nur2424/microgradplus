"""
A small 1D regression problem: fit y = sin(3x) + small noise.

Compares plain SGD vs Adam, both training the same MLP architecture from
the same initialization, to make the optimizer comparison fair and to
demonstrate the optim.py module on a regression (MSE) task rather than
classification.
"""

import random

import matplotlib.pyplot as plt
import numpy as np

from microgradplus.engine import Value
from microgradplus.losses import mse_loss
from microgradplus.nn import MLP
from microgradplus.optim import SGD, Adam
from microgradplus.training import fit, predict

SEED = 0


def make_dataset(n=40):
    rng = np.random.default_rng(SEED)
    xs = np.linspace(-2, 2, n)
    ys = np.sin(3 * xs) + 0.1 * rng.standard_normal(n)
    X = [[float(x)] for x in xs]
    y = [float(v) for v in ys]
    return xs, X, y


def build_model():
    random.seed(SEED)
    return MLP(nin=1, nouts=[8, 8, 1], activation="tanh",
               out_activation="linear", init="xavier")


def main():
    xs, X, y = make_dataset()

    # Build two structurally-identical models from the same seed so the
    # comparison isolates the optimizer, not the initialization.
    random.seed(SEED)
    model_sgd = build_model()
    random.seed(SEED)
    model_adam = build_model()

    opt_sgd = SGD(model_sgd.parameters(), lr=0.05)
    opt_adam = Adam(model_adam.parameters(), lr=0.02)

    print("Training with plain SGD...")
    history_sgd = fit(model_sgd, opt_sgd, mse_loss, X, y,
                       epochs=150, batch_size=None, log_every=50)

    print("\nTraining with Adam...")
    history_adam = fit(model_adam, opt_adam, mse_loss, X, y,
                        epochs=150, batch_size=None, log_every=50)

    # ---- Plot 1 loss curves ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history_sgd, label="SGD (lr=0.05)")
    ax.plot(history_adam, label="Adam (lr=0.02)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("SGD vs Adam fitting y = sin(3x) + noise")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/regression_loss_curves.png", dpi=150)
    plt.close(fig)

    # ---- Plot 2 fitted functions ----
    xs_dense = np.linspace(-2, 2, 200)
    X_dense = [[float(x)] for x in xs_dense]
    preds_sgd = predict(model_sgd, X_dense)
    preds_adam = predict(model_adam, X_dense)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, y, s=15, color="gray", label="training data", alpha=0.6)
    ax.plot(xs_dense, np.sin(3 * xs_dense), "k--", label="true sin(3x)", linewidth=1)
    ax.plot(xs_dense, preds_sgd, label="MLP (SGD)")
    ax.plot(xs_dense, preds_adam, label="MLP (Adam)")
    ax.set_title("Fitted MLP vs ground truth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/regression_fit.png", dpi=150)
    plt.close(fig)

    print("\nSaved plots to examples/regression_loss_curves.png and "
          "examples/regression_fit.png")


if __name__ == "__main__":
    main()
