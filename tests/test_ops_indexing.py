import numpy as np
import torch

from tests.utils import make_tensor, make_torch, tdata, assert_close, assert_grad_close


def test_getitem_slices_forward_backward(rng, device):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = xt[:, 1:, ::2, -3:]
    y = x[:, 1:, ::2, -3:]

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_getitem_integer_index_forward_backward(rng, device):
    x_np = rng.normal(size=(3, 4, 5, 6)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = xt[1, :, 2, :]
    y = x[1, :, 2, :]

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_gather_forward_backward(rng, device):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    dim = int(rng.integers(0, x_np.ndim))

    idx_np = rng.integers(0, x_np.shape[dim], size=x_np.shape, dtype=np.int64)

    xt = make_torch(x_np, requires_grad=True)
    it = torch.tensor(idx_np, dtype=torch.long)

    x = make_tensor(x_np, requires_grad=True, device=device)
    i = make_tensor(idx_np.astype(np.float32), requires_grad=False, device=device)

    yt = xt.gather(dim=dim, index=it)
    y = x.gather(dim, i)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)