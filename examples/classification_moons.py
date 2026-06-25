"""
examples/classification_moons.py

Binary classification on the classic "two moons" dataset.

This demonstrates the full micrograd+ pipeline on a non-trivial,
non-linearly-separable problem: an MLP with ReLU hidden units, He
initialization, a sigmoid output, BCE loss, and Adam.

scikit-learn and matplotlib are used ONLY for data generation and
plotting — every gradient and parameter update is computed by
microgradplus's pure-Python engine.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from microgradplus.engine import Value
from microgradplus.losses import bce_loss
from microgradplus.nn import MLP
from microgradplus.optim import Adam
from microgradplus.training import fit, predict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def main():
    X, y = make_moons(n_samples=200, noise=0.2, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    X_train_list = X_train.tolist()
    y_train_list = [float(v) for v in y_train]

    model = MLP(
        nin=2,
        nouts=[8, 8, 1],
        activation="relu",
        out_activation="sigmoid",
        init="he",
    )
    print(model)
    print(f"parameter count: {len(model.parameters())}")

    optimizer = Adam(model.parameters(), lr=0.05)

    # Full-batch gradient descent: with only 160 training points and a
    # pure-Python engine, mini-batching adds Python-loop overhead without
    # a speed benefit, so we pass the whole training set each epoch.
    history = fit(
        model,
        optimizer,
        bce_loss,
        X_train_list,
        y_train_list,
        epochs=150,
        batch_size=None,
        log_every=25,
    )

    # ---- Test accuracy ----
    test_preds = predict(model, X_test.tolist())
    test_pred_labels = [1 if p > 0.5 else 0 for p in test_preds]
    accuracy = np.mean(np.array(test_pred_labels) == y_test)
    print(f"\ntest accuracy: {accuracy:.3f}")

    # ---- Plot 1: training loss curve ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Training loss — two moons")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/moons_loss_curve.png", dpi=150)
    plt.close(fig)

    # ---- Plot 2: decision boundary ----
    fig, ax = plt.subplots(figsize=(6, 5))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()].tolist()
    grid_preds = predict(model, grid_points)
    zz = np.array(grid_preds).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap="RdBu",
        edgecolors="k", s=30, label="train"
    )
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdBu",
        edgecolors="k", s=60, marker="^", label="test"
    )
    ax.set_title(f"Decision boundary — two moons (test acc = {accuracy:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("examples/moons_decision_boundary.png", dpi=150)
    plt.close(fig)

    print("\nSaved plots to examples/moons_loss_curve.png and "
          "examples/moons_decision_boundary.png")


if __name__ == "__main__":
    main()
