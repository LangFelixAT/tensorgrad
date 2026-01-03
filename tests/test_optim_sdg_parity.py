import numpy as np
import pytest
import torch

from tensorgrad.tensor import Tensor
from tensorgrad.optim import SGD
from tests.utils import make_tensor, to_numpy, assert_close


def run_one_step_sgd(x_np, g_np, *, lr, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
    xt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    xt.grad = torch.tensor(g_np, dtype=torch.float32)

    opt_t = torch.optim.SGD(
        [xt],
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    opt_t.step()

    x = Tensor(np.array(x_np, dtype=np.float32), requires_grad=True, device="cpu")
    x.grad = x.xp().array(g_np, dtype=x.xp().float32) if hasattr(x, "xp") else np.array(g_np, dtype=np.float32)

    opt = SGD([x], lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
    opt.step()

    return xt.detach().cpu().numpy(), to_numpy(x.data)


@pytest.mark.parametrize("cfg", [
    dict(lr=1e-2, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False),
    dict(lr=1e-2, momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=False),
    dict(lr=1e-2, momentum=0.9, dampening=0.1, weight_decay=0.0, nesterov=False),
    dict(lr=1e-2, momentum=0.9, dampening=0.0, weight_decay=1e-3, nesterov=False),
    dict(lr=1e-2, momentum=0.9, dampening=0.0, weight_decay=1e-3, nesterov=True),
])
def test_sgd_single_step_matches_torch(rng, cfg):
    x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
    g_np = rng.normal(size=(2, 3, 4)).astype(np.float32)

    a, b = run_one_step_sgd(x_np, g_np, **cfg)
    assert_close(b, a, atol=1e-6, rtol=1e-5)


def test_sgd_multi_step_matches_torch(rng):
    lr = 1e-2
    momentum = 0.9
    dampening = 0.0
    weight_decay = 1e-3
    nesterov = True

    x0 = rng.normal(size=(3, 4, 5)).astype(np.float32)

    xt = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    opt_t = torch.optim.SGD([xt], lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    x = Tensor(np.array(x0, dtype=np.float32), requires_grad=True, device="cpu")
    opt = SGD([x], lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    for _ in range(5):
        g = rng.normal(size=x0.shape).astype(np.float32)

        xt.grad = torch.tensor(g, dtype=torch.float32)
        opt_t.step()

        x.grad = np.array(g, dtype=np.float32)
        opt.step()

    assert_close(to_numpy(x.data), xt.detach().cpu().numpy(), atol=2e-6, rtol=2e-5)