"""A small synthetic pairing experiment, not a real image/text model.
Run: python contrastive_numpy.py
Requires: numpy. CPU only; fixed seed; independent held-out samples.
"""
import numpy as np

def normalized(z):
    norm = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return z / norm, norm

def objective(x, y, w, u, temperature=0.15):
    a, na = normalized(x @ w)
    b, nb = normalized(y @ u)
    logits = a @ b.T / temperature
    count = len(x)
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)
    p, q = softmax(logits), softmax(logits.T)
    loss = -0.5 * (np.log(np.diag(p)).mean() + np.log(np.diag(q)).mean())
    ds = (p + q.T - 2 * np.eye(count)) / (2 * count)
    da, db = ds @ b / temperature, ds.T @ a / temperature
    dz_a = (da - a * (da * a).sum(axis=1, keepdims=True)) / na
    dz_b = (db - b * (db * b).sum(axis=1, keepdims=True)) / nb
    return loss, x.T @ dz_a, y.T @ dz_b

def recall(x, y, w, u):
    a, _ = normalized(x @ w)
    b, _ = normalized(y @ u)
    scores = a @ b.T
    labels = np.arange(len(x))
    return (scores.argmax(axis=1) == labels).mean(), (scores.argmax(axis=0) == labels).mean()

def main():
    rng = np.random.default_rng(42)
    observation_x = rng.normal(size=(12, 24))
    observation_y = rng.normal(size=(12, 20))
    def sample(n):
        latent = rng.normal(size=(n, 12))
        return (latent @ observation_x + 0.05 * rng.normal(size=(n, 24)),
                latent @ observation_y + 0.05 * rng.normal(size=(n, 20)))
    x, y = sample(512)
    test_x, test_y = sample(128)
    w, u = rng.normal(scale=0.15, size=(24, 16)), rng.normal(scale=0.15, size=(20, 16))
    # Check one analytical derivative against a finite difference.
    _, dw, _ = objective(x[:16], y[:16], w, u)
    epsilon = 1e-5
    plus, minus = w.copy(), w.copy()
    plus[0, 0] += epsilon
    minus[0, 0] -= epsilon
    numerical = (objective(x[:16], y[:16], plus, u)[0] - objective(x[:16], y[:16], minus, u)[0]) / (2 * epsilon)
    assert np.isclose(numerical, dw[0, 0], atol=1e-6, rtol=1e-4), 'Gradient check failed'
    print('Gradient check: passed')
    before = recall(test_x, test_y, w, u)
    print('Held-out Recall@1 before (x->y, y->x):', before)
    for step in range(401):
        ids = rng.choice(len(x), 64, replace=False)
        loss, dw, du = objective(x[ids], y[ids], w, u)
        w -= 0.08 * dw
        u -= 0.08 * du
        if step % 100 == 0:
            print(f'Step {step:3d} | training loss {loss:.4f}')
    after = recall(test_x, test_y, w, u)
    print('Held-out Recall@1 after  (x->y, y->x):', after)
    print('This checks a synthetic learning mechanism, not real multimodal understanding.')

if __name__ == '__main__':
    main()
