"""
optim.py parameter update rules.

Each optimizer wraps a list of Value parameters (typically
model.parameters()) and implements .step(), which mutates .data in
place using .grad, plus .zero_grad().

This is the direct analogue of the
for p in model.parameters(): p.data += -lr * p.grad loop in the
original notebook generalized so the update rule is swappable.
"""


class Optimizer:
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Vanilla stochastic gradient descent: p.data -= lr * p.grad.

    With momentum > 0, accumulates an exponential moving average of
    past gradients (the velocity) and steps in that direction instead,
    which damps oscillations and speeds convergence in narrow valleys.
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self._velocity = [0.0 for _ in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if self.momentum > 0.0:
                self._velocity[i] = self.momentum * self._velocity[i] + p.grad
                p.data -= self.lr * self._velocity[i]
            else:
                p.data -= self.lr * p.grad


class Adam(Optimizer):
    """Adam (Kingma, Ba, 2015).

    Maintains per-parameter running averages of the gradient (m, like
    momentum) and the squared gradient (v, an estimate of the gradient's
    variance), with bias correction for their initialization at zero.
    The effective per-parameter step size is roughly lr * m / sqrt(v),
    which adapts to how noisy/large each parameter's gradient has been.
    """

    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self._m = [0.0 for _ in self.parameters]
        self._v = [0.0 for _ in self.parameters]

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for i, p in enumerate(self.parameters):
            g = p.grad
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g

            m_hat = self._m[i] / (1 - b1 ** self.t)
            v_hat = self._v[i] / (1 - b2 ** self.t)

            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
