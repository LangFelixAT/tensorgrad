import numpy as np
import pytest
import torch

from tensorgrad.tensor import Tensor
from tensorgrad.optim import AdamW
from tests.utils import to_numpy, assert_close


def _make_param(x_np):
    return Tensor(np.array(x_np, dtype=np.float32), requires_grad=True, device="cpu")


def _set_grad(param: Tensor, g_np: np.ndarray):
    param.grad = np.array(g_np, dtype=np.float32)


def _torch_param(x_np):
    return torch.tensor(np.array(x_np, dtype=np.float32), requires_grad=True)


def _torch_set_grad(t: torch.Tensor, g_np: np.ndarray):
    t.grad = torch.tensor(np.array(g_np, dtype=np.float32))


def _make_optim(params, **cfg):
    return AdamW(params, **cfg)


def _make_optim_torch(params, **cfg):
    return torch.optim.AdamW(params, **cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2),
        dict(lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-1),
        dict(lr=1e-3, betas=(0.95, 0.98), eps=1e-8, weight_decay=1e-2),
        dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=1e-2),
    ],
)
def test_adamw_single_step_matches_torch(rng, cfg):
    x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
    g_np = rng.normal(size=(2, 3, 4)).astype(np.float32)

    xt = _torch_param(x_np)
    opt_t = _make_optim_torch([xt], **cfg)
    _torch_set_grad(xt, g_np)
    opt_t.step()

    x = _make_param(x_np)
    opt = _make_optim([x], **cfg)
    _set_grad(x, g_np)
    opt.step()

    assert_close(to_numpy(x.data), xt.detach().cpu().numpy(), atol=2e-6, rtol=2e-5)


def test_adamw_multi_step_matches_torch(rng):
    cfg = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)

    x0 = rng.normal(size=(3, 4, 5)).astype(np.float32)

    xt = _torch_param(x0)
    opt_t = _make_optim_torch([xt], **cfg)

    x = _make_param(x0)
    opt = _make_optim([x], **cfg)

    for _ in range(10):
        g = rng.normal(size=x0.shape).astype(np.float32)

        _torch_set_grad(xt, g)
        opt_t.step()

        _set_grad(x, g)
        opt.step()

    assert_close(to_numpy(x.data), xt.detach().cpu().numpy(), atol=3e-6, rtol=3e-5)


def test_adamw_weight_decay_is_decoupled(rng):
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    wd = 0.1

    x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
    g_np = rng.normal(size=(2, 3, 4)).astype(np.float32)

    xt = _torch_param(x_np)
    opt_t = torch.optim.AdamW([xt], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    _torch_set_grad(xt, g_np)
    opt_t.step()
    x_t = xt.detach().cpu().numpy()

    x = _make_param(x_np)
    opt = _make_optim([x], lr=lr, betas=betas, eps=eps, weight_decay=wd)
    _set_grad(x, g_np)
    opt.step()
    x_y = to_numpy(x.data)

    assert_close(x_y, x_t, atol=2e-6, rtol=2e-5)