import numpy as np
import pytest
import torch.nn.functional as F

from tests.utils import make_tensor, make_torch, tdata, assert_close, assert_grad_close


@pytest.mark.parametrize("op", ["exp", "tanh", "sigmoid", "relu", "neg"])
def test_unary_ops_forward_backward(rng, op, device):
    shape = tuple(int(x) for x in rng.integers(1, 6, size=int(rng.integers(1, 6))))
    x_np = rng.normal(size=shape).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    if op == "exp":
        yt = xt.exp()
        y = x.exp()
    elif op == "tanh":
        yt = xt.tanh()
        y = x.tanh()
    elif op == "sigmoid":
        yt = xt.sigmoid()
        y = x.sigmoid()
    elif op == "relu":
        yt = xt.relu()
        y = x.relu()
    elif op == "neg":
        yt = -xt
        y = -x
    else:
        raise RuntimeError(op)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_log_forward_backward_positive(rng, device):
    shape = tuple(int(x) for x in rng.integers(1, 6, size=int(rng.integers(1, 6))))
    x_np = (rng.random(size=shape).astype(np.float32) + 0.1)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = xt.log()
    y = x.log()

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy())
    assert_grad_close(x, xt)


def test_gelu_tanh_approx_forward_backward(rng, device):
    shape = (2, 3, 4, 5)
    x_np = rng.normal(size=shape).astype(np.float32)

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = F.gelu(xt, approximate="tanh")
    y = x.gelu()

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy(), atol=2e-6, rtol=2e-5)
    assert_grad_close(x, xt, atol=2e-6, rtol=2e-5)