import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tensorgrad.tensor import Tensor
from tests.utils import make_tensor, make_torch, tdata, tgrad, assert_close


def conv_out_size(H, W, kH, kW, sH, sW, dH, dW, pH, pW):
    Hout = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    Wout = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    return Hout, Wout


@pytest.mark.parametrize("kH,kW,sH,sW,dH,dW,pH,pW", [
    (3, 3, 1, 1, 1, 1, 0, 0),
    (3, 3, 2, 2, 1, 1, 1, 1),
    (2, 3, 1, 2, 1, 1, 0, 1),
    (3, 2, 1, 1, 2, 2, 2, 1),
])
def test_im2col_matches_torch_unfold_forward(rng, device, kH, kW, sH, sW, dH, dW, pH, pW):
    N, C, H, W = 2, 3, 7, 8
    x_np = rng.normal(size=(N, C, H, W)).astype(np.float32)

    Hout, Wout = conv_out_size(H, W, kH, kW, sH, sW, dH, dW, pH, pW)
    assert Hout > 0 and Wout > 0

    xt = make_torch(x_np, requires_grad=True)
    x = make_tensor(x_np, requires_grad=True, device=device)

    X_t = F.unfold(
        xt,
        kernel_size=(kH, kW),
        dilation=(dH, dW),
        padding=(pH, pW),
        stride=(sH, sW),
    )

    X = Tensor._im2col(x, kH, kW, sH, sW, dH, dW, Hout, Wout, pH, pW)

    assert_close(tdata(X), X_t.detach().cpu().numpy())


@pytest.mark.parametrize("kH,kW,sH,sW,dH,dW,pH,pW", [
    (3, 3, 1, 1, 1, 1, 1, 1),
    (3, 3, 2, 2, 1, 1, 1, 1),
])
def test_im2col_gemm_conv_forward_backward_matches_torch_unfold(rng, device, kH, kW, sH, sW, dH, dW, pH, pW):
    N, C, H, W = 2, 3, 9, 10
    Cout = 4

    x_np = rng.normal(size=(N, C, H, W)).astype(np.float32)
    w_np = rng.normal(size=(Cout, C * kH * kW)).astype(np.float32)

    Hout, Wout = conv_out_size(H, W, kH, kW, sH, sW, dH, dW, pH, pW)
    assert Hout > 0 and Wout > 0
    L = Hout * Wout

    xt = make_torch(x_np, requires_grad=True)
    wt = torch.tensor(w_np, dtype=torch.float32, requires_grad=True)

    X_t = F.unfold(
        xt,
        kernel_size=(kH, kW),
        dilation=(dH, dW),
        padding=(pH, pW),
        stride=(sH, sW),
    )  # (N, C*kH*kW, L)

    out_t = torch.einsum("oc,ncl->nol", wt, X_t)
    out_t = out_t.reshape(N, Cout, Hout, Wout)

    loss_t = out_t.sum()
    loss_t.backward()

    x = make_tensor(x_np, requires_grad=True, device=device)
    Wg = make_tensor(w_np, requires_grad=True, device=device)

    X = Tensor._im2col(x, kH, kW, sH, sW, dH, dW, Hout, Wout, pH, pW)  # (N, C*kH*kW, L)
    out = Wg @ X
    out = out.reshape(N, Cout, Hout, Wout)

    out.sum().backward()

    assert_close(tdata(out), out_t.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)

    assert x.grad is not None
    assert Wg.grad is not None
    assert_close(tgrad(x), xt.grad.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)
    assert_close(tgrad(Wg), wt.grad.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)