"""
training.py a small fit loop that ties together a model, a loss
function, and an optimizer

Updated to support multi-class models (MLPClassifier) where the model
returns a list of C logits per sample instead of a single Value
"""

import random
from .engine import Value


def fit(model, optimizer, loss_fn, X, y, epochs=100, batch_size=None,
        verbose=True, log_every=10, multiclass=False):
    """Train model on data (X, y)

    Args:
        - model: an MLP / MLPClassifier (anything with .parameters() and
            __call__)

        - optimizer: an Optimizer instance wrapping model.parameters()

        - loss_fn: a function (ypred_list, ytrue_list) -> Value
            For multiclass=True, ypred_list is a list of lists of Values
            (one inner list of C logits per sample)
        
        - X: list of input feature vectors (each a list of floats)

        - y: list of targets. Floats for regression/binary; int class
            indices for multi-class (multiclass=True)
        
        - epochs: number of passes over the full dataset

        - batch_size: if None, full-batch gradient descent, Otherwise
            mini-batch SGD
        
        - verbose: print loss progress

        - log_every: how often (in epochs) to print the loss

        - multiclass: if True, the model returns a list of logits per
            sample (used with categorical_cross_entropy)

    Returns:
        history: list of per-epoch mean loss values
    """
    n = len(X)
    history = []

    for epoch in range(epochs):
        if batch_size is None:
            batches = [(X, y)]
        else:
            idx = list(range(n))
            random.shuffle(idx)
            batches = []
            for start in range(0, n, batch_size):
                bidx = idx[start:start + batch_size]
                batches.append(([X[i] for i in bidx], [y[i] for i in bidx]))

        epoch_loss = 0.0
        total_seen = 0

        for xb, yb in batches:
            if multiclass:
                # Each output is a list of C Value logits
                ypred = [model(x) for x in xb]
            else:
                raw = [model(x) for x in xb]
                ypred = [yp if isinstance(yp, Value) else yp[0] for yp in raw]

            loss = loss_fn(ypred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data * len(xb)
            total_seen += len(xb)

        epoch_loss /= total_seen
        history.append(epoch_loss)

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:>4} | loss = {epoch_loss:.6f}")

    return history


def predict(model, X):
    """Run the model on a list of inputs, returning plain floats or lists"""
    out = []
    for x in X:
        yp = model(x)
        if isinstance(yp, Value):
            out.append(yp.data)
        else:
            out.append([v.data for v in yp])
    return out


def predict_classes(model, X):
    """For MLPClassifier: return predicted class indices"""
    return [model.predict_class(x) for x in X]
