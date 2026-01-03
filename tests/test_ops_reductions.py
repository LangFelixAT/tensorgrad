import numpy as np
import pytest

from tests.utils import make_tensor, make_torch, tdata, assert_close, assert_grad_close


def random_dims(rng, ndim, k=2):
    """Pick up to k distinct dims."""
    k = min(k, ndim)
    dims = sorted({int(rng.integers(0, ndim)) for _ in range(k)})
    return tuple(dims)


@pytest.mark.parametrize("dim_kind", ["none", "single", "multi"])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_forward_backward(rng, device, dim_kind, keepdim):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    if dim_kind == "none":
        yt = xt.sum()
        y = x.sum()
    elif dim_kind == "single":
        dim = int(rng.integers(0, xt.ndim))
        yt = xt.sum(dim=dim, keepdim=keepdim)
        y = x.sum(dim=dim, keepdim=keepdim)
    else:
        dims = random_dims(rng, xt.ndim, k=2)
        yt = xt.sum(dim=dims, keepdim=keepdim)
        y = x.sum(dim=dims, keepdim=keepdim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


@pytest.mark.parametrize("keepdim", [False, True])
def test_mean_forward_backward(rng, device, keepdim):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    dim = int(rng.integers(0, xt.ndim))
    yt = xt.mean(dim=dim, keepdim=keepdim)
    y = x.mean(dim=dim, keepdim=keepdim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


@pytest.mark.parametrize("unbiased", [True, False])
@pytest.mark.parametrize("keepdim", [False, True])
def test_var_forward_backward(rng, device, unbiased, keepdim):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    dim = int(rng.integers(0, xt.ndim))
    yt = xt.var(dim=dim, unbiased=unbiased, keepdim=keepdim)
    y = x.var(dim=dim, unbiased=unbiased, keepdim=keepdim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)
    assert_grad_close(x, xt, atol=5e-6, rtol=5e-5)


@pytest.mark.parametrize("keepdim", [False, True])
def test_logsumexp_forward_backward(rng, device, keepdim):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    dim = int(rng.integers(0, xt.ndim))
    yt = xt.logsumexp(dim=dim, keepdim=keepdim)
    y = x.logsumexp(dim=dim, keepdim=keepdim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy(), atol=3e-6, rtol=3e-5)
    assert_grad_close(x, xt, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("keepdim", [False, True])
def test_max_forward_backward_unique(rng, device, keepdim):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    dim = int(rng.integers(0, x_np.ndim))

    offsets = np.linspace(0.0, 1e-3, x_np.shape[dim], dtype=np.float32)
    shape = [1] * x_np.ndim
    shape[dim] = x_np.shape[dim]
    x_np = x_np + offsets.reshape(shape)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = xt.max(dim=dim, keepdim=keepdim).values
    y = x.max(dim=dim, keepdim=keepdim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)