import numpy as np
import pytest
import torch

from tensorgrad.tensor import Tensor
from tensorgrad.optim import Adam
from tests.utils import to_numpy, assert_close


def _make_param(x_np):
    return Tensor(np.array(x_np, dtype=np.float32), requires_grad=True, device="cpu")


def _set_grad(param: Tensor, g_np: np.ndarray):
    param.grad = np.array(g_np, dtype=np.float32)


def _torch_param(x_np):
    t = torch.tensor(np.array(x_np, dtype=np.float32), requires_grad=True)
    return t


def _torch_set_grad(t: torch.Tensor, g_np: np.ndarray):
    t.grad = torch.tensor(np.array(g_np, dtype=np.float32))


def _make_optim(params, **cfg):
    return Adam(params, **cfg)


def _make_optim_torch(params, **cfg):
    return torch.optim.Adam(params, **cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
        dict(lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2),
        dict(lr=1e-3, betas=(0.95, 0.98), eps=1e-8, weight_decay=0.0),
        dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0),
    ],
)
def test_adam_single_step_matches_torch(rng, cfg):
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


def test_adam_multi_step_matches_torch(rng):
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


@pytest.mark.parametrize("amsgrad", [False, True])
def test_adam_amsgrad_matches_torch_if_supported(rng, amsgrad):
    x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
    g_np = rng.normal(size=(2, 3, 4)).astype(np.float32)

    cfg = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=amsgrad)

    xt = _torch_param(x_np)
    opt_t = _make_optim_torch([xt], **cfg)
    _torch_set_grad(xt, g_np)
    opt_t.step()

    x = _make_param(x_np)
    opt = Adam([x], **cfg)

    _set_grad(x, g_np)
    opt.step()

    assert_close(to_numpy(x.data), xt.detach().cpu().numpy(), atol=2e-6, rtol=2e-5)