import numpy as np
import pytest

from tests.utils import make_tensor, make_torch, tdata, assert_close, assert_grad_close


def test_reshape_forward_backward(rng, device):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = xt.reshape(2, 3, 20)
    y = x.reshape(2, 3, 20)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_transpose_forward_backward(rng, device):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    dim0, dim1 = 1, 3
    yt = xt.transpose(dim0, dim1)
    y = x.transpose(dim0, dim1)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_permute_forward_backward(rng, device):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    dims = (3, 0, 2, 1)
    yt = xt.permute(*dims)
    y = x.permute(*dims)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_unsqueeze_squeeze_dim_roundtrip_backward(rng, device):
    x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    dim = 1
    yt = xt.unsqueeze(dim).squeeze(dim)
    y = x.unsqueeze(dim).squeeze(dim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_squeeze_all_backward(rng, device):
    x_np = rng.normal(size=(1, 2, 1, 3, 1)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = xt.squeeze()
    y = x.squeeze()

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_squeeze_tuple_dims_backward(rng, device):
    x_np = rng.normal(size=(1, 2, 1, 3, 1)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    dims = (0, 2, 4)
    yt = xt.squeeze(dims)
    y = x.squeeze(dims)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_T_property_2d_backward(rng, device):
    x_np = rng.normal(size=(4, 7)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = xt.T
    y = x.T

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)