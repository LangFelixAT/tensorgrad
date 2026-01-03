import numpy as np
import pytest

from tensorgrad.tensor import no_grad
from tests.utils import make_tensor, assert_close


def test_no_grad_disables_tracking(rng, device):
    x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
    x = make_tensor(x_np, requires_grad=True, device=device)

    with no_grad():
        y = x * 2 + 1

    assert y.requires_grad is False, "no_grad() should disable graph construction"


def test_requires_grad_propagation(rng, device):
    x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)

    x = make_tensor(x_np, requires_grad=False, device=device)
    y = make_tensor(x_np, requires_grad=True, device=device)

    z1 = x + x
    z2 = x + y

    assert z1.requires_grad is False
    assert z2.requires_grad is True


def test_zero_grad_clears_existing_grad(rng, device):
    x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
    x = make_tensor(x_np, requires_grad=True, device=device)

    (x * x).sum().backward()
    assert x.grad is not None

    x.zero_grad()
    assert_close(x.grad, np.zeros_like(x_np, dtype=np.float32))


def test_backward_accumulates(rng, device):
    x_np = rng.normal(size=(2, 3, 4)).astype(np.float32)
    x = make_tensor(x_np, requires_grad=True, device=device)

    (x * 2).sum().backward()
    g1 = x.grad.copy()

    (x * 3).sum().backward()
    g2 = x.grad.copy()

    assert_close(g2 - g1, 3.0 * np.ones_like(x_np, dtype=np.float32))


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
def test_backward_default_grad_for_non_scalar_matches_ones(shape, rng, device):
    x_np = rng.normal(size=shape).astype(np.float32)
    x = make_tensor(x_np, requires_grad=True, device=device)

    y = x * 2
    y.backward()

    assert_close(x.grad, 2.0 * np.ones_like(x_np, dtype=np.float32))