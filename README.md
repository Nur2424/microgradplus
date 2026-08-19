# micrograd+

A tiny scalar-valued autograd engine and neural network library, built from
scratch in pure Python

The core library (`microgradplus/`) has **zero dependencies**. Every gradient
is computed by hand written reverse mode automatic differentiation over a
dynamically built computation graph no NumPy, no PyTorch. The goal is that
nothing inside `loss.backward()` is mysterious.

```
git clone <https://github.com/Nur2424/microgradplus>
cd microgradplus
pip install -e .
```

## What's in here

```
microgradplus/
├── engine.py     # the Value class: scalar autograd
├── nn.py         # Neuron, Layer, MLP
├── losses.py     # MSE, binary cross-entropy, hinge loss
├── optim.py      # SGD (+momentum), Adam
├── training.py   # fit() / predict() training loop
└── viz.py        # computation graph visualization (graphviz)

tests/            # 44 tests every gradient checked vs PyTorch + finite differences
examples/         # two end-to-end demos with generated plots
notebooks/        # the original derivation notebook (study notes)
```

## Quick start

```python
import random
from microgradplus.nn import MLP
from microgradplus.losses import mse_loss
from microgradplus.optim import Adam
from microgradplus.training import fit, predict

random.seed(42)
model = MLP(nin=2, nouts=[16, 16, 1], activation="tanh", out_activation="linear")

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [0.0, 1.0, 1.0, 0.0]  # XOR

opt = Adam(model.parameters(), lr=0.05)
fit(model, opt, mse_loss, X, y, epochs=200, log_every=50)

print(predict(model, X))
```

## The engine

Everything is built on one class, `Value`, which wraps a scalar and
remembers the operation and operands that produced it:

```python
from microgradplus.engine import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
L = d.tanh()

L.backward()
print(a.grad, b.grad, c.grad)
```

Each operator (`+`, `*`, `**`, `exp`, `log`) and activation (`tanh`,
`sigmoid`, `relu`, `leaky_relu`) attaches a closure its **local
derivative rule** to the output node. `.backward()` builds a reverse
topological ordering of the graph via DFS and calls each node's local rule
in reverse order, accumulating gradients with `+=` (the multivariate chain
rule across multiple paths the same accumulation property that motivates
`optimizer.zero_grad()`).

| micrograd+ concept | PyTorch equivalent |
|---|---|
| `Value` | `torch.Tensor` with `requires_grad=True` |
| `_prev`, `_op` | the autograd graph (`grad_fn` chain) |
| `_backward` closure | `Function.backward` |
| `Value.backward()` | `Tensor.backward()` |
| `optimizer.zero_grad()` | `for p in params: p.grad = 0.0` |
| `SGD.step()` / `Adam.step()` | `optimizer.step()` |

## Design choices

**Activations and initialization are paired deliberately.** `nn.py` exposes
`activation="tanh"|"sigmoid"|"relu"|"leaky_relu"|"linear"` and
`init="uniform"|"xavier"|"he"`. Xavier/Glorot initialization keeps
activation variance roughly constant across layers for saturating
activations (tanh, sigmoid); He initialization compensates for ReLU zeroing
out about half its inputs. The original micrograd used
`uniform(-1, 1)` everywhere, which works for a 4-point toy dataset but
saturates tanh units as networks grow.

**Losses are framed as maximum-likelihood objectives**, following the
"negative-log-likelihood-to-loss" correspondence: Gaussian noise on a
continuous target → MSE; a Bernoulli label → binary cross-entropy;
hinge loss for the perceptron/SVM-style margin objective. For sigmoid + BCE,
the gradient with respect to the pre-sigmoid logit is exactly
`(prediction - target) / N` — the single most-reused gradient in
classification, and `tests/test_losses.py` checks this identity directly.

**Optimizers are separated from the model**, mirroring
`torch.optim`. `SGD` supports momentum (an exponential moving average of
past gradients, damping oscillations in narrow valleys); `Adam` additionally
tracks a per-parameter second-moment estimate so the effective step size
adapts to how noisy each parameter's gradient has been.

## Correctness

Every operation and activation in `engine.py` is tested two ways:

1. **Against PyTorch's autograd** on identical expressions (`test_engine.py`,
   `test_losses.py`).
2. **Against finite-difference numerical derivatives**
   (`test_finite_difference_matches_backward`).

```
pip install -e ".[dev]"
pytest -v
```

44/44 tests pass.

## Examples

### 1. Binary classification two moons (`examples/classification_moons.py`)

An MLP with ReLU hidden units (He init) and a sigmoid output, trained with
Adam and binary cross-entropy on `sklearn.datasets.make_moons`.
scikit-learn/matplotlib are used only for data generation and plotting —
every gradient and update comes from `microgradplus`.

Test accuracy: **0.95**

| Training loss | Decision boundary |
|---|---|
| ![loss](examples/moons_loss_curve.png) | ![boundary](examples/moons_decision_boundary.png) |

### 2. Regression fitting `sin(3x)` (`examples/regression_sgd_vs_adam.py`)

Two structurally identical MLPs (same seed, same architecture), one trained
with plain SGD and one with Adam, on a noisy `y = sin(3x)` dataset — isolating
the effect of the optimizer.

| Loss curves (log scale) | Fitted curves |
|---|---|
| ![loss](examples/regression_loss_curves.png) | ![fit](examples/regression_fit.png) |

Run either example yourself:

```
pip install -e ".[examples]"
python examples/classification_moons.py
python examples/regression_sgd_vs_adam.py
```

## The original notebook

`notebooks/01_micrograd_from_scratch.ipynb` is where this all started: a
from-scratch derivation of backpropagation, building `Value` up incrementally
(forward pass → manual gradients → closures → topological sort), with every
analytical gradient checked against finite differences and PyTorch along the
way. The library in `microgradplus/` is the same engine, refactored into a
reusable package and extended with more losses, activations, initialization
schemes, and optimizers.

## What this is not

This is a learning/portfolio project, not a production framework. There's no
vectorization (everything is scalar `Value`s an MLP forward pass is O(number
of weights) Python function calls), no GPU support, and no batching beyond
simple Python loops. The engineering that PyTorch adds on top vectorized
tensors, fused kernels, an iterative (not recursive) topological sort, gradient
checkpointing is real and substantial. The point here is that the
*mathematical content* underneath all of that is exactly this.

## Possible next steps

- Vectorize `Value` to operate on small NumPy arrays instead of scalars
  (a natural "micrograd → nanograd" step).
- Add softmax + categorical cross-entropy for multi-class classification.
- Add dropout / L2 weight decay.
- Add a learning-rate scheduler.

## Acknowledgements

Built while working through Andrej Karpathy's
[*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html),
starting from his [micrograd](https://github.com/karpathy/micrograd).
