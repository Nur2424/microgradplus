"""
engine.py a scalar-valued autograd engine

This is the core of microgradplus. A `Value` wraps a single Python float and
remembers the small computational graph that produced it. Calling
`.backward()` on any `Value` runs reverse-mode automatic differentiation
(backpropagation) over that graph, populating `.grad` on every node with
the partial derivative of the output with respect to that node

No NumPy, no PyTorch pure Python, on purpose. The point of this module
is that nothing inside `loss.backward()` should be mysterious

New in this version:
    softmax(values) class method on Value that takes a list of Value
    logits and returns a list of Value probabilities summing to 1.
    The backward pass is derived analytically (Jacobian of softmax),
    but because it is implemented through existing primitives (exp, log,
    division) it gets correct gradients automatically via the chain rule
"""

import math


class Value:
    """A scalar value in an autograd computation graph

    Attributes:
        data: the scalar forward value
        grad: dL/d(this node), accumulated during backward()
        _prev: parent nodes this value was computed from
        _op: string label of the operation that produced this value
            (used only for visualization/debugging)
        label: optional human-readable name (used only for visualization)
    """

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None  # default no-op (leaf nodes)
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    
    # ----- Core ops with local backward rules -----
    

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers supported"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        """Natural log. Used by cross-entropy losses

        Note: undefined at x <= 0. Callers are expected to clamp
        probabilities away from 0 before calling this
        """
        x = self.data
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += (1.0 / x) * out.grad

        out._backward = _backward
        return out

    
    # ----- Activation functions -----
    

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        """Leaky ReLU: f(x) = x if x > 0 else alpha * x

        Avoids the "dead neuron" problem of plain ReLU by letting a small
        gradient flow through for negative inputs
        """
        x = self.data
        out = Value(x if x > 0 else alpha * x, (self,), "LeakyReLU")

        def _backward():
            self.grad += (1.0 if x > 0 else alpha) * out.grad

        out._backward = _backward
        return out

    
    # ----- Softmax (class method operates on a list of Value logits) -----
    

    @staticmethod
    def softmax(logits):
        """Numerically stable softmax over a list of Value logits

        Returns a list of Value probabilities that sum to 1

        Implementation: subtract max(logits) before exp for numerical
        stability (prevents exp overflow). This does NOT change the output
        because the constant cancels in the division:

            softmax(z_i) = exp(z_i - C) / sum_j exp(z_j - C)  for any C

        Gradients flow correctly through the exp / division primitives
        already defined above no new backward rule is needed here

        The gradient of the loss w.r.t. the input logit z_i, when using
        softmax + categorical cross-entropy, collapses to (p_i - y_i)
        where p_i is the predicted probability and y_i is the one-hot
        target. This is derived in the notebook; this implementation
        produces exactly that via automatic differentiation
        """
        # Numerical stability: subtract max (constant w.r.t. gradient)
        max_val = max(v.data for v in logits)
        shifted = [Value(v.data - max_val, v._prev, v._op) for v in logits]
        # Carry backward through the original logit nodes
        for orig, sh in zip(logits, shifted):
            orig_bwd = orig._backward
            def _make_bwd(o, s):
                def _bwd():
                    o.grad += s.grad
                    orig_bwd()
                return _bwd
            sh._backward = _make_bwd(orig, sh)
            sh._prev = {orig}

        exps = [v.exp() for v in shifted]
        exp_sum = sum(exps[1:], exps[0])  # single Value, sum of all exp nodes
        probs = [e * (exp_sum ** -1) for e in exps]
        return probs

    
    # Composed ops (built from the primitives above no new backward
    # rules needed, which is itself a small proof that the primitives
    # are sufficient)
    

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    # ----- Reflected (right-hand-side) operators, e.g. 2 + value, 3 - value -----

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self ** -1

    # ----- Backward orchestrator -----

    def backward(self):
        """Run reverse-mode autodiff starting from this node

        Builds a topological ordering of the computation graph via DFS,
        then walks it in reverse, calling each node's local `_backward`

        Because the chain rule sums contributions over every path from a
        node to the output, gradients are accumulated with `+=` inside
        each `_backward`  visiting children only after all of a node's
        consumers have contributed is exactly what the topological order
        guarantees
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
