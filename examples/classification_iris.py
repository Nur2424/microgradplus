"""
classification_iris.py — 3-class classification on the Iris dataset.

What this demonstrates:
  - MLPClassifier with ReLU hidden layers (He init)
  - Categorical cross-entropy loss (softmax applied inside the loss)
  - Adam optimizer
  - Decision regions visualization for the first two features

scikit-learn and matplotlib are used only for data and plotting.
Every gradient and every weight update comes from microgradplus.
"""

import random
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from microgradplus.nn import MLPClassifier
from microgradplus.losses import categorical_cross_entropy
from microgradplus.optim import Adam
from microgradplus.training import fit, predict_classes

random.seed(42)
np.random.seed(42)

# ── Data ──────────────────────────────────────────────────────────────────────
iris = load_iris()
X_raw, y_raw = iris.data[:, :2], iris.target   # use first 2 features for 2-D plot

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).tolist()
y_list = y_raw.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_list, test_size=0.2, random_state=42, stratify=y_list
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = MLPClassifier(nin=2, hidden=[16, 16], n_classes=3,
                      activation="relu", init="he")
opt = Adam(model.parameters(), lr=0.02)

# ── Training ──────────────────────────────────────────────────────────────────
history = fit(
    model, opt, categorical_cross_entropy,
    X_train, y_train,
    epochs=300, batch_size=16,
    log_every=50, multiclass=True,
)

# ── Accuracy ──────────────────────────────────────────────────────────────────
y_pred_train = predict_classes(model, X_train)
y_pred_test  = predict_classes(model, X_test)

train_acc = sum(p == t for p, t in zip(y_pred_train, y_train)) / len(y_train)
test_acc  = sum(p == t for p, t in zip(y_pred_test,  y_test))  / len(y_test)
print(f"\nTrain accuracy: {train_acc:.3f}   Test accuracy: {test_acc:.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Loss curve
axes[0].plot(history, color="#2563EB", linewidth=1.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Mean CCE Loss")
axes[0].set_title("Training Loss — Iris 3-class")
axes[0].grid(True, alpha=0.3)

# Decision boundary
h = 0.05
x0_vals = np.array([x[0] for x in X_scaled])
x1_vals = np.array([x[1] for x in X_scaled])
x0_min, x0_max = x0_vals.min() - 0.5, x0_vals.max() + 0.5
x1_min, x1_max = x1_vals.min() - 0.5, x1_vals.max() + 0.5
xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h),
                     np.arange(x1_min, x1_max, h))
grid = [[float(a), float(b)] for a, b in zip(xx.ravel(), yy.ravel())]
Z = np.array(predict_classes(model, grid)).reshape(xx.shape)

colors_bg = ["#DBEAFE", "#D1FAE5", "#FEF3C7"]
colors_pt = ["#1D4ED8", "#047857", "#B45309"]
labels    = iris.target_names

axes[1].contourf(xx, yy, Z, alpha=0.4,
                 colors=colors_bg, levels=[-0.5, 0.5, 1.5, 2.5])
for cls in range(3):
    pts = [(x[0], x[1]) for x, l in zip(X_scaled, y_list) if l == cls]
    xs_cls = [p[0] for p in pts]
    ys_cls = [p[1] for p in pts]
    axes[1].scatter(xs_cls, ys_cls, color=colors_pt[cls],
                    label=labels[cls], s=25, edgecolors="white", linewidths=0.5)

axes[1].set_xlabel("Feature 1 (scaled)")
axes[1].set_ylabel("Feature 2 (scaled)")
axes[1].set_title(f"Decision Boundary  |  Test acc: {test_acc:.0%}")
axes[1].legend()
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
outpath = "examples/iris_softmax_cce.png"
plt.savefig(outpath, dpi=150)
print(f"Saved → {outpath}")
