import numpy as np

from tests.utils import make_tensor, make_torch, tdata, assert_close, assert_grad_close


def test_matmul_2d_forward_backward(rng, device):
    a_np = rng.normal(size=(6, 10)).astype(np.float32)
    b_np = rng.normal(size=(10, 4)).astype(np.float32)

    at = make_torch(a_np, requires_grad=True)
    bt = make_torch(b_np, requires_grad=True)
    a = make_tensor(a_np, requires_grad=True, device=device)
    b = make_tensor(b_np, requires_grad=True, device=device)

    yt = at @ bt
    y = a @ b

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy(), atol=3e-6, rtol=3e-5)
    assert_grad_close(a, at, atol=3e-6, rtol=3e-5)
    assert_grad_close(b, bt, atol=3e-6, rtol=3e-5)


def test_matmul_batched_same_batch_forward_backward(rng, device):
    a_np = rng.normal(size=(2, 3, 6, 10)).astype(np.float32)  # B1,B2,M,K
    b_np = rng.normal(size=(2, 3, 10, 5)).astype(np.float32)  # B1,B2,K,N

    at = make_torch(a_np, requires_grad=True)
    bt = make_torch(b_np, requires_grad=True)
    a = make_tensor(a_np, requires_grad=True, device=device)
    b = make_tensor(b_np, requires_grad=True, device=device)

    yt = at @ bt
    y = a @ b

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)
    assert_grad_close(a, at, atol=5e-6, rtol=5e-5)
    assert_grad_close(b, bt, atol=5e-6, rtol=5e-5)


def test_matmul_batched_broadcast_batch_forward_backward(rng, device):
    a_np = rng.normal(size=(4, 6, 10)).astype(np.float32)     # B,M,K
    b_np = rng.normal(size=(1, 10, 5)).astype(np.float32)     # 1,K,N

    at = make_torch(a_np, requires_grad=True)
    bt = make_torch(b_np, requires_grad=True)
    a = make_tensor(a_np, requires_grad=True, device=device)
    b = make_tensor(b_np, requires_grad=True, device=device)

    yt = at @ bt
    y = a @ b

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)
    assert_grad_close(a, at, atol=5e-6, rtol=5e-5)
    assert_grad_close(b, bt, atol=5e-6, rtol=5e-5)


def test_matmul_vector_matrix_cases_forward_backward(rng,device):
    v_np = rng.normal(size=(10,)).astype(np.float32)
    m_np = rng.normal(size=(10, 5)).astype(np.float32)

    vt = make_torch(v_np, requires_grad=True)
    mt = make_torch(m_np, requires_grad=True)

    v = make_tensor(v_np, requires_grad=True, device=device)
    m = make_tensor(m_np, requires_grad=True, device=device)

    yt = vt @ mt
    y = v @ m

    yt.sum().backward()
    y.sum().backward()

    assert_close(tdata(y), yt.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)
    assert_grad_close(v, vt, atol=5e-6, rtol=5e-5)
    assert_grad_close(m, mt, atol=5e-6, rtol=5e-5)

    m2_np = rng.normal(size=(6, 10)).astype(np.float32)
    v2_np = rng.normal(size=(10,)).astype(np.float32)

    m2t = make_torch(m2_np, requires_grad=True)
    v2t = make_torch(v2_np, requires_grad=True)

    m2 = make_tensor(m2_np, requires_grad=True, device=device)
    v2 = make_tensor(v2_np, requires_grad=True, device=device)

    yt2 = m2t @ v2t
    y2 = m2 @ v2

    yt2.sum().backward()
    y2.sum().backward()

    assert_close(tdata(y2), yt2.detach().cpu().numpy(), atol=5e-6, rtol=5e-5)
    assert_grad_close(m2, m2t, atol=5e-6, rtol=5e-5)
    assert_grad_close(v2, v2t, atol=5e-6, rtol=5e-5)