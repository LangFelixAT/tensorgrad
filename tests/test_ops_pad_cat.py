import numpy as np
import torch
import torch.nn.functional as F

from tensorgrad.tensor import Tensor
from tests.utils import make_tensor, make_torch, tdata, assert_close, assert_grad_close


def test_pad2d_forward_backward(rng, device):
    x_np = rng.normal(size=(2, 3, 8, 9)).astype(np.float32)
    padding = (2, 1, 3, 0)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = F.pad(xt, padding, mode="constant", value=0.0)
    y = x.pad2d(padding)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_cat_forward_backward_dim1(rng, device):
    a_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    b_np = rng.normal(size=(2, 7, 4, 5)).astype(np.float32)
    dim = 1

    at = make_torch(a_np, requires_grad=True)
    bt = make_torch(b_np, requires_grad=True)

    a = make_tensor(a_np, requires_grad=True, device=device)
    b = make_tensor(b_np, requires_grad=True, device=device)

    yt = torch.cat([at, bt], dim=dim)
    y = Tensor._cat([a, b], dim=dim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(a, at)
    assert_grad_close(b, bt)


def test_cat_forward_backward_negative_dim(rng, device):
    a_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    b_np = rng.normal(size=(2, 3, 4, 6)).astype(np.float32)
    dim = -1

    at = make_torch(a_np, requires_grad=True)
    bt = make_torch(b_np, requires_grad=True)

    a = make_tensor(a_np, requires_grad=True, device=device)
    b = make_tensor(b_np, requires_grad=True, device=device)

    yt = torch.cat([at, bt], dim=dim)
    y = Tensor._cat([a, b], dim=dim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(a, at)
    assert_grad_close(b, bt)