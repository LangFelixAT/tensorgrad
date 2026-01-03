import numpy as np
import torch

from tensorgrad.tensor import Tensor

ATOL = 1e-6
RTOL = 1e-5

def _is_cupy(x):
    return x.__class__.__module__.startswith("cupy")

def to_numpy(x):
    if _is_cupy(x):
        import cupy as cp
        return cp.asnumpy(x)
    return np.asarray(x)

def tdata(t: Tensor):
    return to_numpy(t.data)

def tgrad(t: Tensor):
    return None if t.grad is None else to_numpy(t.grad)

def make_tensor(x_np: np.ndarray, requires_grad: bool = True, device: str = "cpu") -> Tensor:
    return Tensor(np.asarray(x_np, dtype=np.float32), requires_grad=requires_grad, device=device)

def make_torch(x_np: np.ndarray, requires_grad: bool = True) -> torch.Tensor:
    return torch.tensor(np.asarray(x_np, dtype=np.float32), requires_grad=requires_grad)

def assert_close(a, b, atol=ATOL, rtol=RTOL):
    a = to_numpy(a)
    b = to_numpy(b)
    assert np.allclose(a, b, atol=atol, rtol=rtol), f"max|diff|={np.max(np.abs(a-b))}"

def assert_grad_close(t: Tensor, tt: torch.Tensor, atol=ATOL, rtol=RTOL):
    assert t.grad is not None, "Your Tensor.grad is None"
    assert tt.grad is not None, "Torch grad is None"
    assert_close(tgrad(t), tt.grad.detach().cpu().numpy(), atol=atol, rtol=rtol)