"""
ann_engine.py
-------------
A small Artificial Neural Network written from scratch with NumPy.

Architecture : Input -> [Dense + ReLU] x N -> Dense(1) + Sigmoid
Loss         : Weighted Binary Cross-Entropy (failures are rare, so we boost them)
Optimizer    : Adam, mini-batch
Regulariser  : L2 weight decay + early stopping
"""
import numpy as np


# ----------------------------------------------------------------------------
# Activation functions
# ----------------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -60, 60)              # stops overflow inside exp()
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------
# Neural network
# ----------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.003, l2=1e-4, seed=42):
        self.layer_sizes = list(layer_sizes)          # e.g. [9, 32, 16, 1]
        self.lr = learning_rate
        self.l2 = l2
        self.rng = np.random.default_rng(seed)

        self.W, self.b = [], []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # He initialisation works best with ReLU
            self.W.append(self.rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))

        # Adam optimiser memory
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    # ---------------------------------------------------------------- forward
    def _forward(self, X):
        activations, zs = [X], []
        a = X
        last = len(self.W) - 1
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            a = sigmoid(z) if i == last else relu(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    # --------------------------------------------------------------- backward
    def _backward(self, activations, zs, y, sample_w):
        m = y.shape[0]
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.W)

        # Sigmoid + cross-entropy gives a very clean error term: (p - y)
        delta = (activations[-1] - y) * sample_w / m
        for i in reversed(range(len(self.W))):
            grads_W[i] = activations[i].T @ delta + self.l2 * self.W[i]
            grads_b[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * relu_grad(zs[i - 1])
        return grads_W, grads_b

    # ------------------------------------------------------------------- Adam
    def _adam_update(self, grads_W, grads_b, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            for param, grad, m, v in (
                (self.W, grads_W, self.mW, self.vW),
                (self.b, grads_b, self.mb, self.vb),
            ):
                m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
                m_hat = m[i] / (1 - beta1 ** self.t)
                v_hat = v[i] / (1 - beta2 ** self.t)
                param[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------- loss
    @staticmethod
    def _loss(p, y, sample_w):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(np.mean(bce * sample_w))

    # -------------------------------------------------------------------- fit
    def fit(self, X, y, X_val, y_val, epochs=80, batch_size=64,
            pos_weight=5.0, patience=20, progress_callback=None):
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        y_val = np.asarray(y_val, dtype=float).reshape(-1, 1)
        w_train = np.where(y == 1, pos_weight, 1.0)
        w_val = np.where(y_val == 1, pos_weight, 1.0)

        history = {"loss": [], "val_loss": []}
        best_val, best_weights, wait = np.inf, None, 0
        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)               # shuffle every epoch
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                acts, zs = self._forward(X[idx])
                gW, gb = self._backward(acts, zs, y[idx], w_train[idx])
                self._adam_update(gW, gb)

            loss = self._loss(self._forward(X)[0][-1], y, w_train)
            val_loss = self._loss(self._forward(X_val)[0][-1], y_val, w_val)
            history["loss"].append(loss)
            history["val_loss"].append(val_loss)

            if progress_callback:
                progress_callback(epoch, epochs, loss, val_loss)

            # early stopping: keep the best weights seen on validation data
            if val_loss < best_val - 1e-5:
                best_val, wait = val_loss, 0
                best_weights = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_weights is not None:
            self.W, self.b = best_weights
        return history

    # ---------------------------------------------------------------- predict
    def predict_proba(self, X):
        return self._forward(np.asarray(X, dtype=float))[0][-1].ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------ save / load
    def count_parameters(self):
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def to_dict(self):
        return {"layer_sizes": self.layer_sizes, "W": self.W, "b": self.b}

    @classmethod
    def from_dict(cls, data):
        net = cls(data["layer_sizes"])
        net.W, net.b = data["W"], data["b"]
        return net