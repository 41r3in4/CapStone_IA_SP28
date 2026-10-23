import numpy as np


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))


class MLP:
    """One-hidden-layer binary MLP; NumPy implementation for inspectable training."""
    def __init__(self, dim, hidden=24, seed=42):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": rng.normal(0, np.sqrt(2 / dim), (dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1 / np.sqrt(hidden), hidden),
            "b2": np.zeros(1),
        }

    def encode(self, X):
        return np.maximum(0, X @ self.p["W1"] + self.p["b1"])

    def predict_proba(self, X):
        return sigmoid(self.encode(X) @ self.p["W2"] + self.p["b2"][0])

    def snapshot(self):
        return {k: v.copy() for k, v in self.p.items()}

    def loss_grad(self, X, y, anchor=None, fisher=None, ewc=0.0, teacher=None, reg=0.0):
        a = X @ self.p["W1"] + self.p["b1"]
        h = np.maximum(a, 0)
        logits = h @ self.p["W2"] + self.p["b2"][0]
        prob = sigmoid(logits)
        loss = np.mean(np.logaddexp(0, logits) - y * logits)
        delta = (prob - y) / len(y)
        if teacher is not None and reg:
            # BCE distillation only on malicious samples correctly detected by teacher.
            old = sigmoid(np.maximum(0, X @ teacher["W1"] + teacher["b1"]) @ teacher["W2"] + teacher["b2"][0])
            mask = (y == 1) & (old >= 0.5)
            loss += reg * np.mean(mask * (np.logaddexp(0, logits) - old * logits))
            delta += reg * mask * (prob - old) / len(y)
        dh = delta[:, None] * self.p["W2"][None, :] * (a > 0)
        grad = {"W1": X.T @ dh, "b1": dh.sum(axis=0), "W2": h.T @ delta, "b2": np.array([delta.sum()])}
        if anchor is not None and fisher is not None and ewc:
            for k in self.p:
                diff = self.p[k] - anchor[k]
                loss += 0.5 * ewc * np.sum(fisher[k] * diff**2)
                grad[k] += ewc * fisher[k] * diff
        return float(loss), grad

    def fit(self, X, y, epochs, lr, batch, rng, **kwargs):
        for _ in range(epochs):
            idx = rng.permutation(len(y))
            for start in range(0, len(idx), batch):
                chosen = idx[start:start+batch]
                loss, grad = self.loss_grad(X[chosen], y[chosen], **kwargs)
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; inspect data and learning rate")
                norm = np.sqrt(sum(np.sum(g*g) for g in grad.values()))
                for k in self.p:
                    self.p[k] -= lr * grad[k] * min(1.0, 10.0 / max(norm, 1e-12))

    def empirical_fisher(self, X, y, limit, rng):
        result = {k: np.zeros_like(v) for k, v in self.p.items()}
        idx = rng.choice(len(y), size=min(limit, len(y)), replace=False)
        for i in idx:
            _, g = self.loss_grad(X[i:i+1], y[i:i+1])
            for k in result:
                result[k] += g[k] ** 2 / len(idx)
        return result
