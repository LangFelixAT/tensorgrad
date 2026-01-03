import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tensorgrad.tensor import Tensor
from tensorgrad.optim import SGD, AdamW
from tests.utils import to_numpy, assert_close


def _mse_loss_torch(pred, target):
    return ((pred - target) ** 2).mean()


def _mse_loss(pred: Tensor, target: Tensor):
    return ((pred - target) ** 2).mean()


def test_linear_regression_sgd_step_by_step_parity(rng):
    N, Din, Dout = 64, 10, 3
    X = rng.normal(size=(N, Din)).astype(np.float32)
    w0 = rng.normal(size=(Din, Dout)).astype(np.float32)
    b0 = rng.normal(size=(Dout,)).astype(np.float32)

    y_true = (X @ (w0 * 0.5) + b0 * 0.25).astype(np.float32)

    lr = 1e-1
    steps = 20

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y_true, dtype=torch.float32)
    wt = torch.tensor(w0, dtype=torch.float32, requires_grad=True)
    bt = torch.tensor(b0, dtype=torch.float32, requires_grad=True)
    opt_t = torch.optim.SGD([wt, bt], lr=lr, momentum=0.0)

    Xy = Tensor(X, requires_grad=False, device="cpu")
    yy = Tensor(y_true, requires_grad=False, device="cpu")
    w = Tensor(w0, requires_grad=True, device="cpu")
    b = Tensor(b0, requires_grad=True, device="cpu")
    opt = SGD([w, b], lr=lr, momentum=0.0)

    for _ in range(steps):
        opt_t.zero_grad()
        pred_t = Xt @ wt + bt
        loss_t = _mse_loss_torch(pred_t, yt)
        loss_t.backward()
        opt_t.step()

        w.zero_grad()
        b.zero_grad()
        pred = Xy @ w + b
        loss = _mse_loss(pred, yy)
        loss.backward()
        opt.step()

        assert_close(to_numpy(w.data), wt.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)
        assert_close(to_numpy(b.data), bt.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)


def test_tiny_mlp_adamw_loss_decreases(rng):
    N = 128
    X = rng.uniform(-1, 1, size=(N, 2)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.int64)

    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    h = 16
    W1 = rng.normal(scale=0.3, size=(2, h)).astype(np.float32)
    b1 = np.zeros((h,), dtype=np.float32)
    W2 = rng.normal(scale=0.3, size=(h, 2)).astype(np.float32)
    b2 = np.zeros((2,), dtype=np.float32)

    lr = 2e-2
    wd = 1e-2
    steps = 200

    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)

    W1t = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
    b1t = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
    W2t = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
    b2t = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

    opt_t = torch.optim.AdamW([W1t, b1t, W2t, b2t], lr=lr, weight_decay=wd)

    Xy = Tensor(X, requires_grad=False, device="cpu")
    Yy = Tensor(Y, requires_grad=False, device="cpu")

    W1y = Tensor(W1, requires_grad=True, device="cpu")
    b1y = Tensor(b1, requires_grad=True, device="cpu")
    W2y = Tensor(W2, requires_grad=True, device="cpu")
    b2y = Tensor(b2, requires_grad=True, device="cpu")

    opt = AdamW([W1y, b1y, W2y, b2y], lr=lr, weight_decay=wd)

    def forward_torch():
        h1 = torch.relu(Xt @ W1t + b1t)
        logits = h1 @ W2t + b2t
        return logits

    def forward_yours():
        h1 = (Xy @ W1y + b1y).relu()
        logits = h1 @ W2y + b2y
        return logits

    with torch.no_grad():
        loss0_t = ((forward_torch() - Yt) ** 2).mean().item()
    loss0_y = float(to_numpy(((forward_yours() - Yy) ** 2).mean().data))

    for _ in range(steps):
        opt_t.zero_grad()
        logits_t = forward_torch()
        loss_t = ((logits_t - Yt) ** 2).mean()
        loss_t.backward()
        opt_t.step()

        for p in [W1y, b1y, W2y, b2y]:
            p.zero_grad()
        logits = forward_yours()
        loss = ((logits - Yy) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        loss1_t = ((forward_torch() - Yt) ** 2).mean().item()
    loss1_y = float(to_numpy(((forward_yours() - Yy) ** 2).mean().data))

    assert loss1_t < 0.7 * loss0_t
    assert loss1_y < 0.7 * loss0_y

    assert abs(loss1_y - loss1_t) / max(loss1_t, 1e-6) < 0.35