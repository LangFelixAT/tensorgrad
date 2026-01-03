import numpy as np
import pytest

from tests.utils import make_tensor, make_torch, tdata, assert_close, assert_grad_close


def make_broadcastable_shapes(rng, min_nd=1, max_nd=5, min_size=1, max_size=6):
    a_nd = int(rng.integers(min_nd, max_nd + 1))
    b_nd = int(rng.integers(min_nd, max_nd + 1))
    nd = max(a_nd, b_nd)

    a = []
    b = []
    for _ in range(nd):
        s = int(rng.integers(min_size, max_size + 1))
        r = float(rng.random())
        if r < 0.33:
            a.append(1); b.append(s)
        elif r < 0.66:
            a.append(s); b.append(1)
        else:
            a.append(s); b.append(s)

    a = tuple(a[nd - a_nd :])
    b = tuple(b[nd - b_nd :])
    return a, b


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_binary_ops_broadcast_forward_backward(rng, op, device):
    for _ in range(10):
        a_shape, b_shape = make_broadcastable_shapes(rng)
        a_np = rng.normal(size=a_shape).astype(np.float32)
        b_np = rng.normal(size=b_shape).astype(np.float32)

        if op == "div":
            b_np = b_np + 0.3

        at = make_torch(a_np, requires_grad=True)
        bt = make_torch(b_np, requires_grad=True)
        a = make_tensor(a_np, requires_grad=True, device=device)
        b = make_tensor(b_np, requires_grad=True, device=device)

        if op == "add":
            yt = at + bt
            y = a + b
        elif op == "sub":
            yt = at - bt
            y = a - b
        elif op == "mul":
            yt = at * bt
            y = a * b
        elif op == "div":
            yt = at / bt
            y = a / b
        else:
            raise RuntimeError(op)

        yt.sum().backward()
        y.sum().backward()

        assert_close(tdata(y), yt.detach().cpu().numpy())
        assert_grad_close(a, at)
        assert_grad_close(b, bt)


def test_pow_integer_exponent_forward_backward(rng, device):
    for _ in range(5):
        x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
        p = int(rng.integers(2, 6))

        xt = make_torch(x_np, requires_grad=True)
        x = make_tensor(x_np, requires_grad=True, device=device)

        yt = xt ** p
        y = x ** p

        yt.sum().backward()
        y.sum().backward()

        assert_close(tdata(y), yt.detach().cpu().numpy())
        assert_grad_close(x, xt)


def test_pow_float_exponent_positive_base_forward_backward(rng, device):
    for _ in range(5):
        x_np = (rng.random(size=(2, 3, 4, 5)).astype(np.float32) + 0.2)
        p = float(rng.random() * 2.5 + 0.5)

        xt = make_torch(x_np, requires_grad=True)
        x = make_tensor(x_np, requires_grad=True, device=device)

        yt = xt ** p
        y = x ** p

        yt.sum().backward()
        y.sum().backward()

        assert_close(tdata(y), yt.detach().cpu().numpy(), atol=2e-6, rtol=2e-5)
        assert_grad_close(x, xt, atol=2e-6, rtol=2e-5)


def test_rpow_forward_backward(rng, device):
    for _ in range(5):
        x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

        xt = make_torch(x_np, requires_grad=True)
        x = make_tensor(x_np, requires_grad=True, device=device)

        yt = 2.0 ** xt
        y = 2.0 ** x

        yt.sum().backward()
        y.sum().backward()

        assert_close(tdata(y), yt.detach().cpu().numpy(), atol=2e-6, rtol=2e-5)
        assert_grad_close(x, xt, atol=2e-6, rtol=2e-5)