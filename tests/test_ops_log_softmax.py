import numpy as np

from tests.utils import make_tensor, make_torch, tdata, assert_close, assert_grad_close


def test_log_softmax_forward_backward(rng, device):
    x_np = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    dim = int(rng.integers(0, x_np.ndim))

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    yt = xt.log_softmax(dim=dim)
    y = x.log_softmax(dim=dim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy(), atol=3e-6, rtol=3e-5)
    assert_grad_close(x, xt, atol=3e-6, rtol=3e-5)


def test_log_softmax_numerical_stability_large_values(device):
    x_np = np.array(
        [
            [[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]],
            [[50.0, 0.0, -50.0], [20.0, 10.0, 0.0]],
        ],
        dtype=np.float32,
    )

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    dim = 2
    yt = xt.log_softmax(dim=dim)
    y = x.log_softmax(dim=dim)

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy(), atol=3e-6, rtol=3e-5)
    assert_grad_close(x, xt, atol=3e-6, rtol=3e-5)