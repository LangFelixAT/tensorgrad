import numpy as np
import pytest
import importlib

from tensorgrad.tensor import Tensor
from tests.utils import assert_close, to_numpy


def test_factories_shapes_dtypes_requires_grad_cpu(device):
    z = Tensor.zeros(2, 3, 4, requires_grad=False, device=device)
    o = Tensor.ones(2, 3, 4, requires_grad=False, device=device)
    r = Tensor.randn(2, 3, 4, requires_grad=False, device=device)
    i = Tensor.randint(0, 10, size=(2, 3, 4), requires_grad=False, device=device)

    assert z.shape == (2, 3, 4)
    assert o.shape == (2, 3, 4)
    assert r.shape == (2, 3, 4)
    assert i.shape == (2, 3, 4)

    assert z.dtype == np.float32
    assert o.dtype == np.float32
    assert r.dtype == np.float32
    assert i.dtype == np.float32

    assert z.requires_grad is False
    assert o.requires_grad is False
    assert r.requires_grad is False
    assert i.requires_grad is False

    assert_close(to_numpy(z.data), np.zeros((2, 3, 4), dtype=np.float32))
    assert_close(to_numpy(o.data), np.ones((2, 3, 4), dtype=np.float32))


def test_to_cpu_is_identity(device):
    x = Tensor.randn(2, 3, 4, requires_grad=False, device=device)
    y = x.to("cpu")
    assert_close(to_numpy(y.data), to_numpy(x.data))


def test_xp_cpu_returns_numpy(device):
    x = Tensor.randn(2, 3, 4, requires_grad=False, device=device)
    xp = x.xp()
    assert xp.__name__ == "numpy" if device == "cpu" else "cupy"


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="cupy not installed")
def test_to_cuda_roundtrip():
    import cupy as cp

    x = Tensor.randn(2, 3, 4, requires_grad=False, device="cpu")
    xc = x.to("cuda")
    assert xc.xp().__name__.startswith("cupy")
    assert isinstance(xc.data, cp.ndarray)

    back = xc.to("cpu")
    assert back.xp().__name__ == "numpy"
    assert_close(to_numpy(back.data), to_numpy(x.data), atol=1e-6, rtol=1e-5)