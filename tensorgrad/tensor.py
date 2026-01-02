from typing import Any, Iterable, Optional, Literal, Sequence, Tuple, Union

import numpy as np
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

def _is_cupy_array(x: Any) -> bool:
    """
    Return whether ``x`` is a CuPy ndarray.

    This is safe when CuPy is not installed: it short-circuits on ``_HAS_CUPY``.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        True if CuPy is available and ``x`` is an instance of ``cupy.ndarray``,
        False otherwise.

    Notes
    -----
    This checks only for ``cupy.ndarray``. If you want to treat *any* object
    implementing the CUDA Array Interface as "GPU-backed", you may extend
    this to check for the ``__cuda_array_interface__`` attribute as well.
    """
    return _HAS_CUPY and hasattr(cp, "ndarray") and isinstance(x, cp.ndarray)

_DeviceStr = Literal["cpu", "cuda"]
def _normalize_device(device: Optional[Union[str, _DeviceStr]]) -> Optional[_DeviceStr]:
    """
    Normalize a device specifier to 'cpu', 'cuda', or None.

    Parameters
    ----------
    device : {None, 'cpu', 'cuda', str}
        Device specifier. If a string starts with 'cuda' (e.g. 'cuda', 'cuda:0'),
        it is normalized to 'cuda'. 'cpu' is preserved. None is returned as None.

    Returns
    -------
    {'cpu', 'cuda', None}
        Normalized device identifier.

    Raises
    ------
    ValueError
        If ``device`` is a string that is neither 'cpu' nor startswith 'cuda'.

    Examples
    --------
    >>> _normalize_device(None)
    >>> _normalize_device('cpu')
    'cpu'
    >>> _normalize_device('cuda')
    'cuda'
    >>> _normalize_device('cuda:1')
    'cuda'
    >>> _normalize_device('gpu')
    Traceback (most recent call last):
        ...
    ValueError: Unknown device spec: 'gpu'
    """
    if device is None:
        return None
    if isinstance(device, str):
        dev = device.lower()
        if dev.startswith("cuda"):
            return "cuda"
        if dev == "cpu":
            return "cpu"
    raise ValueError(f"Unknown device spec: {device!r}")

_grad_enabled = True
"""bool: Global flag indicating whether automatic differentiation is enabled.

This flag is toggled by the :class:``no_grad`` context manager.
When ``_grad_enabled`` is ``False``, operations on tensors will not
be tracked for gradient computation.
"""

class no_grad:
    """
    Context manager that temporarily disables gradient computation.

    When entering this context, operations on tensors will not be tracked
    for autograd, meaning no computational graph is built and no gradients
    will be computed in subsequent backward passes.

    Examples
    --------
    >>> with no_grad():
    ...     y = model(x)   # no gradients tracked
    >>> # Outside the context, gradients are tracked again.

    Notes
    -----
    - This mirrors the behavior of ``torch.no_grad()`` in PyTorch.
    - It is safe to nest ``no_grad`` contexts; the previous state of
      ``_grad_enabled`` is restored upon exit.
    """
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self.prev

class Tensor:
    """
    A multi-dimensional array with automatic differentiation.

    This class wraps a NumPy or CuPy array (selected per instance) and
    records a computation graph when gradient tracking is enabled.
    Backpropagation is triggered via :meth:``backward``.

    Notes
    -----
    - Backend is chosen per tensor: CPU uses NumPy, CUDA uses CuPy.
    - DType is normalized to ``float32`` on construction.
    - Gradient storage ``.grad`` is allocated eagerly ``zeros_like(data)`` only if
      ``requires_grad`` is True *and* global grad mode is enabled.
    - Operations use the instance's backend (``self.backend``) to remain
      device-agnostic.
    """
    def __init__(
        self,
        data: Any,
        _prev: Iterable["Tensor"] = (),
        requires_grad: bool = False,
        device: Optional[str] = None,
    ) -> None:
        """
        Construct a tensor from array-like data, selecting NumPy or CuPy as backend.

        Parameters
        ----------
        data : Any
            Array-like input (e.g., Python list/tuple, ``numpy.ndarray``,
            or ``cupy.ndarray``). If ``device`` is not provided, the backend
            is inferred from ``data``: CuPy if it is a CuPy array (and CuPy is
            available), otherwise NumPy. The data is converted to ``float32``.
        _prev : Iterable[Tensor], optional
            Internal: parent tensors that produced this tensor in the
            computation graph. Used to build the topological order for
            backpropagation. End users should not set this.
        requires_grad : bool, default False
            If True (and global grad mode is enabled), this tensor will
            track operations and accumulate gradients into ``.grad`` during
            :meth:``backward``. If False, no gradient is tracked for this tensor.
        device : {'cpu', 'cuda', 'cuda:0', ...} or None, optional
            Desired device. If 'cuda'/* or 'cuda:idx' is requested but CuPy is
            unavailable, a ``RuntimeError`` is raised. If None, the device is
            inferred from ``data`` (CuPy → 'cuda', otherwise 'cpu').

        Attributes
        ----------
        data : numpy.ndarray or cupy.ndarray
            The underlying mutable array in ``float32``.
        grad : same as ``data`` or None
            Gradient buffer. Allocated as zeros_like(data) when
            ``requires_grad`` is True and global grad mode is enabled;
            otherwise None.
        backend : module
            Either ``numpy`` (CPU) or ``cupy`` (CUDA) for all ops on this tensor.
        requires_grad : bool
            Effective flag after applying global grad mode.
        _prev : set[Tensor]
            Internal set of parent tensors for autograd traversal.

        Raises
        ------
        RuntimeError
            If ``device`` requests CUDA but CuPy is not installed/available.

        Examples
        --------
        >>> Tensor([1, 2, 3])                        # CPU by default
        >>> Tensor(np.ones((2,3), np.float64))       # cast to float32
        >>> Tensor(cp.ones((2,3)), device='cuda')    # CUDA (if CuPy available)
        >>> Tensor([1,2,3], requires_grad=True)
        """
        dev = _normalize_device(device)

        if dev == "cuda":
            if not _HAS_CUPY:
                raise RuntimeError("CUDA requested but CuPy is not installed/available.")
            backend = cp
            data = cp.asarray(data, dtype=cp.float32)
        elif dev == "cpu":
            backend = np
            data = np.asarray(data, dtype=np.float32)
        else:
            if _is_cupy_array(data):
                backend = cp
                data = data.astype(cp.float32)
            elif isinstance(data, np.ndarray):
                backend = np
                data = data.astype(np.float32)
            else:
                backend = np
                data = np.asarray(data, dtype=np.float32)

        self.backend = backend
        self.data = data
        self.requires_grad = bool(requires_grad) and _grad_enabled
        self.grad = self.backend.zeros_like(self.data) if self.requires_grad else None

        self._backward = lambda: None
        self._prev = set(_prev)

    @property
    def shape(self) -> Tuple[int, ...]:
        """tuple of int: The tensor's shape."""
        return self.data.shape

    @property
    def dtype(self) -> Union[np.dtype, str]:
        """numpy.dtype: The data type of the tensor."""
        return self.data.dtype

    @property
    def ndim(self) -> int:
        """int: The number of dimensions of the tensor."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """int: Total number of elements in the tensor."""
        return self.data.size

    @property
    def T(self) -> "Tensor":
        """Tensor: Transposed view of the tensor."""
        axes = tuple(reversed(range(self.ndim)))
        return self.permute(*axes)

    def __add__(self, other: Union["Tensor", Any]) -> "Tensor":
        """
        Elementwise addition with NumPy-style broadcasting.

        Parameters
        ----------
        other : Tensor or array-like
            Value to add. If not a ``Tensor``, it is converted to a tensor on the
            same backend and dtype as ``self`` (via ``_ensure_tensor``).

        Returns
        -------
        Tensor
            The result of ``self + other``. ``requires_grad`` is True if either
            operand requires gradients.

        Notes
        -----
        - Supports NumPy-style broadcasting.
        - Gradients:
        ``dL/dself = unbroadcast(out.grad, self.shape)`` and
        ``dL/dother = unbroadcast(out.grad, other.shape)``.

        Examples
        --------
        >>> a = Tensor([1, 2, 3], requires_grad=True)
        >>> b = Tensor([4, 5, 6])
        >>> c = a + b
        >>> c.shape
        (3,)
        """
        other = Tensor._ensure_tensor(other, self.backend)

        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), requires_grad=requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, Tensor._unbroadcast(out.grad, self.data.shape))
            Tensor._accumulate_grad(other, Tensor._unbroadcast(out.grad, other.data.shape))
        out._backward = _backward

        return out

    def __sub__(self, other: Union["Tensor", Any]) -> "Tensor":
        """Elementwise subtraction; implemented as ``self + (-other)``."""
        return self + (-other)

    def __mul__(self, other: Union["Tensor", Any]) -> "Tensor":
        """
        Elementwise multiplication with NumPy-style broadcasting.

        Parameters
        ----------
        other : Tensor or array-like
            Value to multiply. If not a ``Tensor``, it is converted to a tensor
            on the same backend and dtype as ``self`` (via ``_ensure_tensor``).

        Returns
        -------
        Tensor
            The result of ``self * other``. ``requires_grad`` is True if either
            operand requires gradients.

        Notes
        -----
        - Supports NumPy-style broadcasting.
        - Gradients:
        ``dL/dself = unbroadcast(other.data * out.grad, self.shape)``,
        ``dL/dother = unbroadcast(self.data * out.grad, other.shape)``.
        - If either input contains NaNs or Infs, the result follows backend
        (NumPy/CuPy) semantics.

        Examples
        --------
        >>> a = Tensor([1., 2., 3.], requires_grad=True)
        >>> b = Tensor([4., 5., 6.])
        >>> c = a * b
        >>> c.shape
        (3,)
        """
        other = Tensor._ensure_tensor(other, self.backend)

        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), requires_grad=requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, Tensor._unbroadcast(other.data * out.grad, self.data.shape))
            Tensor._accumulate_grad(other, Tensor._unbroadcast(self.data * out.grad, other.data.shape))
        out._backward = _backward

        return out

    def __truediv__(self, other: Union["Tensor", Any]) -> "Tensor":
        """Elementwise division; implemented as ``self * (other ** -1)``."""
        return self * other**-1

    def __pow__(self, other: Union["Tensor", Any]) -> "Tensor":
        """
        Elementwise power with NumPy-style broadcasting.

        Computes ``self ** other`` elementwise. Both base and exponent may be
        tensors (or array-like) and are broadcast against each other.

        Parameters
        ----------
        other : Tensor or array-like
            Exponent(s). If not a ``Tensor``, it is converted to a tensor on the
            same backend as ``self`` (via ``_ensure_tensor``).

        Returns
        -------
        Tensor
            The result of ``self ** other``. ``requires_grad`` is True if either
            operand requires gradients.

        Notes
        -----
        Gradients (per element), with broadcasting handled by ``_unbroadcast``:
        - dL/dself  = ``other * self ** (other - 1) * out.grad``
        - dL/dother = ``(self ** other) * log(self) * out.grad``

        Caveats
        -------
        - For non-positive bases (``self <= 0``), the derivative wrt ``other`` uses
        ``log(self)`` and will be NaN/Inf in real-valued dtype. This matches
        NumPy/CuPy semantics.
        - When both base and exponent are 0 (``0 ** 0``), the value/gradient is
        backend-defined (usually 1.0) and the gradients are undefined.
        - If desired, numerical warnings can be silenced with the backend's
        error-state context (e.g., ``numpy.errstate`` / ``cupy.errstate``).

        Examples
        --------
        >>> a = Tensor([2., 3., 4.], requires_grad=True)
        >>> b = Tensor([3., 2., 0.5], requires_grad=True)
        >>> c = a ** b
        >>> c.shape
        (3,)
        """
        other = Tensor._ensure_tensor(other, self.backend)

        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data ** other.data, (self, other),requires_grad=requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, Tensor._unbroadcast((other.data * self.data**(other.data - 1)) * out.grad, self.data.shape))
            Tensor._accumulate_grad(other, Tensor._unbroadcast((out.data * self.backend.log(self.data)) * out.grad, other.data.shape))
        out._backward = _backward

        return out

    def __matmul__(self, other: Union["Tensor", Any]) -> "Tensor":
        """
        Matrix multiply (supports batched matmul) with NumPy/CuPy semantics.

        Computes ``self @ other`` using the backend's ``matmul``:
        - If both inputs are 2D: standard matrix multiply ``(m,k)@(k,n)->(m,n)``.
        - If inputs are >=3D: performs **batched** matmul with broadcasting over
        leading batch dimensions (NumPy-style), contracting the last two dims:
        ``(..., m, k) @ (..., k, n) -> (..., m, n)``.

        Parameters
        ----------
        other : Tensor or array-like
            Right-hand operand. If not a ``Tensor``, it is converted to a tensor
            on the same backend as ``self`` (via ``_ensure_tensor``).

        Returns
        -------
        Tensor
            Result of the matrix product. ``requires_grad`` is True if either
            operand requires gradients.

        Notes
        -----
        - Batch dimensions (all but the last two) broadcast per NumPy/CuPy rules.
        - Gradients:
            - ``dL/dself = unbroadcast( out.grad @ swapaxes(other, -1, -2), self.shape )``
            - ``dL/dother = unbroadcast( swapaxes(self, -1, -2) @ out.grad, other.shape )``
        where ``swapaxes(x, -1, -2)`` is the matrix transpose for the last two dims.
        - Any broadcasting in batch dims is reduced back with ``_unbroadcast``.

        Examples
        --------
        2D:
        >>> a = Tensor.randn(2, 3, requires_grad=True)
        >>> b = Tensor.randn(3, 4, requires_grad=True)
        >>> y = a @ b
        >>> y.shape
        (2, 4)

        Batched:
        >>> a = Tensor.randn(5, 2, 3, requires_grad=True)   # batch=5
        >>> b = Tensor.randn(5, 3, 4, requires_grad=True)
        >>> y = a @ b
        >>> y.shape
        (5, 2, 4)
        """
        other = Tensor._ensure_tensor(other, self.backend)

        requires_grad = self.requires_grad or other.requires_grad
        out_data = self.backend.matmul(self.data, other.data)
        out = Tensor(out_data, (self, other), requires_grad=requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, Tensor._unbroadcast(self.backend.matmul(out.grad, self.backend.swapaxes(other.data, -1, -2)), self.data.shape))
            Tensor._accumulate_grad(other, Tensor._unbroadcast(self.backend.matmul(self.backend.swapaxes(self.data, -1, -2), out.grad), other.data.shape))
        out._backward = _backward

        return out

    def __neg__(self) -> "Tensor":
        """Elementwise negation (returns ``-self``)."""
        return self * -1

    def __radd__(self, other: Union["Tensor", Any]) -> "Tensor":
        """Right-hand addition: ``other + self``."""
        return self + other

    def __rsub__(self, other: Union["Tensor", Any]) -> "Tensor":
        """Right-hand subtraction: ``other - self``."""
        return Tensor._ensure_tensor(other, self.backend) - self

    def __rmul__(self, other: Union["Tensor", Any]) -> "Tensor":
        """Right-hand multiplication: ``other * self``."""
        return self * other

    def __rtruediv__(self, other: Union["Tensor", Any]) -> "Tensor":
        """Right-hand division: ``other / self``."""
        return Tensor._ensure_tensor(other, self.backend) / self

    def __rpow__(self, other: Union["Tensor", Any]) -> "Tensor":
        """Right-hand power: ``other ** self``."""
        return Tensor._ensure_tensor(other, self.backend) ** self

    def __getitem__(self, idx: Union[int, slice, tuple]) -> "Tensor":
        """
        Index or slice into the tensor (NumPy-style).

        Supports integer, slice, and tuple indexing (e.g. ``x[0]``, ``x[:, 1:3]``).
        Returns a new tensor containing the selected elements.

        Parameters
        ----------
        idx : int, slice, or tuple
            Index or slice specification, following NumPy/CuPy semantics.

        Returns
        -------
        Tensor
            Returns a new tensor containing the selected elements.

        Notes
        -----
        During backpropagation, the gradient of the output is *scattered back*
        into the original tensor's shape, filling zeros for elements that were
        not selected.

        Examples
        --------
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        >>> y = x[0]
        >>> y.shape
        (3,)
        """
        out_data = self.data[idx]
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            grad = out.grad
            grad_full = self.backend.zeros_like(self.data)
            grad_full[idx] = grad
            Tensor._accumulate_grad(self, grad_full)

        out._backward = _backward
        return out

    def max(
        self, 
        dim: Optional[int] = None,
        keepdim: bool = False,
    ) -> "Tensor":
        """
        Compute the maximum of elements along a dimension.

        Reduces the tensor by taking the maximum value either over all elements
        or along the specified axis. Mirrors NumPy/PyTorch semantics.

        Parameters
        ----------
        dim : int, optional
            The dimension to reduce. If ``None``, computes the global maximum
            over all elements.
        keepdim : bool, default=False
            If True, retains reduced dimensions with length 1, allowing
            broadcasting in later operations.

        Returns
        -------
        Tensor
            The maximum value(s) along the given dimension. If ``dim`` is None,
            a scalar tensor is returned.

        Notes
        -----
        **Gradient behavior:**  
        - Gradients are distributed equally among all elements that achieve the
        maximum value (important when there are ties).  
        - The gradient of each maximum element is ``out.grad / count_max``, where
        ``count_max`` is the number of elements equal to the max value.

        Examples
        --------
        >>> x = Tensor([[1, 3, 2],
        ...             [4, 0, 5]], requires_grad=True)
        >>> y = x.max(dim=1)
        >>> y.data
        array([3., 5.], dtype=float32)
        >>> y.backward()
        >>> x.grad
        tensor([[0., 1., 0.],
                [0., 0., 1.]])
        """
        out_data = self.backend.max(self.data, axis=dim, keepdims=keepdim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            od = out_data if (dim is None or keepdim) else self.backend.expand_dims(out.data, axis=dim)  # need to expand out_data to make it broadcastable with self.data
            mask = (self.data == od).astype(self.data.dtype)
            count = self.backend.sum(mask, axis=dim, keepdims=True)
            grad = (mask / count) * out.grad
            if not keepdim and dim is not None:
                grad = Tensor._expand_like(grad, self.data.shape, dim)
            Tensor._accumulate_grad(self, grad)
        out._backward = _backward

        return out

    def sum(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
    ) -> "Tensor":
        """
        Sum of elements along a dimension.

        Reduces the tensor by summing over all elements or along the specified
        dimension(s). Mirrors NumPy/PyTorch semantics.

        Parameters
        ----------
        dim : int or tuple of int, optional
            Dimension(s) to reduce. If ``None``, computes the global sum.
        keepdim : bool, default=False
            If True, retains reduced dimensions with length 1.

        Returns
        -------
        Tensor
            The summed value(s). If ``dim`` is None, a scalar tensor is returned.

        Notes
        -----
        **Gradient behavior:**  
        The upstream gradient is broadcast back to the input shape. For a global
        sum, the gradient is a tensor of ones (same shape as input) multiplied by
        ``out.grad``. For reductions with ``keepdim=False``, the gradient is
        expanded along the reduced axes.

        Examples
        --------
        >>> x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
        >>> y = x.sum(dim=1)      # shape (2,)
        >>> y.backward(np.array([1., 10.], dtype=np.float32))
        >>> x.grad
        tensor([[ 1.,  1.],
                [10., 10.]])
        """
        out_data = self.backend.sum(self.data, axis=dim, keepdims=keepdim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            grad = out.grad
            if not keepdim and dim is not None:
                grad = Tensor._expand_like(grad, self.data.shape, dim)
            Tensor._accumulate_grad(self, grad)
        out._backward = _backward

        return out

    def mean(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
    ) -> "Tensor":
        """
        Compute the mean of elements along a dimension.

        Averages the tensor elements either globally or along the specified
        dimension(s). Behaves like ``numpy.mean`` or ``torch.mean``.

        Parameters
        ----------
        dim : int or tuple of int, optional
            Dimension(s) to reduce. If ``None``, computes the global mean.
        keepdim : bool, default=False
            If True, retains reduced dimensions with length 1.

        Returns
        -------
        Tensor
            The mean value(s). If ``dim`` is None, returns a scalar tensor.

        Notes
        -----
        The gradient of the mean divides the upstream gradient equally among
        all contributing elements, i.e.  
        ``∂mean/∂x = 1/N`` where ``N`` is the number of elements reduced.

        Examples
        --------
        >>> x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
        >>> y = x.mean()
        >>> y.backward()
        >>> x.grad
        tensor([[0.25, 0.25],
                [0.25, 0.25]])
        """
        out = self.sum(dim=dim, keepdim=keepdim)
        if dim is None:
            divisor = self.data.size
        else:
            divisor = self.data.shape[dim] if isinstance(dim, int) else self.backend.prod([self.data.shape[d] for d in dim])

        return out / self.backend.array(divisor, dtype=self.data.dtype)

    def var(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
        unbiased: bool = True,
    ) -> "Tensor":
        """
        Compute the variance of elements along a dimension.

        Estimates the variance as the average of squared deviations from the mean.
        Supports both population and unbiased (sample) variance, similar to
        ``numpy.var`` and ``torch.var``.

        Parameters
        ----------
        dim : int or tuple of int, optional
            Dimension(s) to reduce. If ``None``, computes the variance of all
            elements in the tensor.
        keepdim : bool, default=False
            If True, retains reduced dimensions with length 1.
        unbiased : bool, default=True
            If True, uses the unbiased estimator (divides by ``N - 1``).  
            If False, divides by ``N``.

        Returns
        -------
        Tensor
            The variance value(s). If ``dim`` is None, a scalar tensor is returned.

        Notes
        -----
        **Mathematical definition:**
        \[
        \text{Var}(x) = \frac{1}{N - δ} \sum_i (x_i - \bar{x})^2
        \]
        where \( δ = 1 \) if ``unbiased=True`` and 0 otherwise.

        **Gradient behavior:**
        - The gradient propagates through both the subtraction and squaring steps.
        - The scaling by \(1/N\) or \(1/(N-1)\) is automatically handled by the
        autograd engine.

        Examples
        --------
        >>> x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
        >>> y = x.var()
        >>> y.backward()
        >>> x.grad
        tensor([[-1.5, -0.5],
                [ 0.5,  1.5]])
        """
        mean = self.mean(dim=dim, keepdim=True)
        sq_diff = (self - mean) ** 2
        out = sq_diff.sum(dim=dim, keepdim=keepdim)

        if dim is None:
            count = self.data.size
        else:
            count = self.data.shape[dim] if isinstance(dim, int) else self.backend.prod([self.data.shape[d] for d in dim])

        divisor = count - 1 if unbiased and count > 1 else count

        return out / self.backend.array(divisor, dtype=self.data.dtype)

    def reshape(
        self,
        *shape: int,
    ) -> "Tensor":
        """
        Return a tensor with the same data but a new shape.

        Parameters
        ----------
        shape : int
            The desired shape (variadic). Must be compatible with the number
            of elements. Follows NumPy/CuPy semantics, so at most one dimension
            may be ``-1`` to infer its size.

        Returns
        -------
        Tensor
            A tensor with ``shape``. May be a view or a copy depending on the
            backend’s ``reshape`` semantics.

        Notes
        -----
        Backpropagation reshapes the upstream gradient back to the original
        input shape.

        Examples
        --------
        >>> x = Tensor.randn(2, 3, requires_grad=True)
        >>> y = x.reshape(3, 2)
        >>> y.shape
        (3, 2)
        """
        out_data = self.data.reshape(*shape)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, out.grad.reshape(self.data.shape))
        out._backward = _backward

        return out

    def transpose(
        self,
        dim0: int,
        dim1: int,
    ) -> "Tensor":
        """
        Swap two dimensions of the tensor.

        Parameters
        ----------
        dim0 : int
            First dimension to swap.
        dim1 : int
            Second dimension to swap.

        Returns
        -------
        Tensor
            A tensor with ``dim0`` and ``dim1`` swapped. Equivalent to
            ``permute`` with those two axes exchanged.

        Examples
        --------
        >>> x = Tensor.randn(2, 3, 4)
        >>> y = x.transpose(1, 2)
        >>> y.shape
        (2, 4, 3)
        """
        dims = list(range(self.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]

        return self.permute(*dims)

    def permute(
        self,
        *dims: int,
    ) -> "Tensor":
        """
        Permute (reorder) the tensor’s dimensions.

        Parameters
        ----------
        dims : int
            A permutation of the axes. If omitted, the axes are reversed
            (same as calling ``transpose()`` with no arguments in NumPy/CuPy).

        Returns
        -------
        Tensor
            A view (or copy, backend-dependent) with axes reordered according to
            ``dims``.

        Notes
        -----
        - ``dims`` must be a reordering of ``range(self.ndim)``.
        - Backpropagation applies the inverse permutation to the upstream
        gradient (computed via ``np.argsort(dims)``). If ``dims`` is omitted,
        both forward and backward use the default axis reversal.

        Examples
        --------
        >>> x = Tensor.randn(2, 3, 4, requires_grad=True)
        >>> y = x.permute(0, 2, 1)   # (N, W, H)
        >>> y.shape
        (2, 4, 3)
        """
        out_data = self.data.transpose(*dims)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            reverse_dims = np.argsort(dims) if dims else None
            Tensor._accumulate_grad(self, out.grad.transpose(*reverse_dims))
        out._backward = _backward

        return out

    def squeeze(
        self,
        dim: Optional[int] = None,
    ) -> "Tensor":
        """
        Remove dimensions of size 1 from the tensor shape.

        Parameters
        ----------
        dim : int, optional
            If specified, only removes the given dimension if its size is 1.  
            If ``None``, all singleton dimensions are removed.

        Returns
        -------
        Tensor
            A tensor with the same data but fewer dimensions.

        Notes
        -----
        - Behaves like ``numpy.squeeze`` and ``torch.squeeze``.
        - During backpropagation, the gradient is reshaped to match the
        original input shape.

        Examples
        --------
        >>> x = Tensor.randn(1, 3, 1, 5)
        >>> x.shape
        (1, 3, 1, 5)
        >>> y = x.squeeze()
        >>> y.shape
        (3, 5)
        >>> z = x.squeeze(2)
        >>> z.shape
        (1, 3, 5)
        """
        out_data = self.backend.squeeze(self.data, dim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, out.grad.reshape(self.shape))
        out._backward = _backward

        return out

    def unsqueeze(
        self,
        dim: int,
    ) -> "Tensor":
        """
        Insert a new dimension of size 1 at the specified position.

        Parameters
        ----------
        dim : int
            The index at which to insert the new dimension.  
            Must be in the range ``[-self.ndim - 1, self.ndim + 1)``.

        Returns
        -------
        Tensor
            A tensor with one additional dimension of size 1.

        Notes
        -----
        - Equivalent to ``numpy.expand_dims`` or ``torch.unsqueeze``.
        - During backpropagation, the gradient is squeezed along the inserted
        axis to restore the original shape.

        Examples
        --------
        >>> x = Tensor.randn(3, 4)
        >>> x.shape
        (3, 4)
        >>> y = x.unsqueeze(0)
        >>> y.shape
        (1, 3, 4)
        >>> z = x.unsqueeze(-1)
        >>> z.shape
        (3, 4, 1)
        """
        out_data = self.backend.expand_dims(self.data, axis=dim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, self.backend.squeeze(out.grad, axis=dim))

        out._backward = _backward

        return out

    def gather(
        self,
        dim: int,
        index: "Tensor",
    ) -> "Tensor":
        """
        Gather values along an axis using integer indices (NumPy/PyTorch-style).

        Selects elements from ``self`` along dimension ``dim`` using ``index``.
        The output shape matches ``index.shape``; for every position in
        ``index``, the value is taken from ``self`` at the same coordinates on
        all other axes and at the indexed position along ``dim``.

        Parameters
        ----------
        dim : int
            Axis along which to gather. Must be a valid dimension of ``self``.
        index : Tensor
            Integer index tensor (same backend/device as ``self``). For
            NumPy/CuPy-style ``take_along_axis``, ``index`` must have the same
            shape as the desired output; all dimensions except ``dim`` should
            match the corresponding dimensions of ``self``. Values must be in
            ``[0, self.shape[dim])``.

        Returns
        -------
        Tensor
            Gathered tensor with shape equal to ``index.shape``.

        Notes
        -----
        - This is a wrapper over the backend's ``take_along_axis``.
        - **Gradients**: only propagate to ``self`` (not to ``index``). In the
        backward pass, gradients are *scattered back* to ``self`` using
        ``put_along_axis``. If indices repeat, their gradients are summed.
        - Behavior and constraints mirror ``numpy.take_along_axis`` /
        ``torch.gather`` (shape rules differ slightly between NumPy and PyTorch;
        here we follow NumPy's requirement that ``index.shape`` equals output
        shape).

        Examples
        --------
        >>> x = Tensor([[10, 11, 12],
        ...             [20, 21, 22]], requires_grad=True)  # shape (2, 3)
        >>> idx = Tensor([[2, 0], [1, 1]])                  # shape (2, 2)
        >>> y = x.gather(dim=1, index=idx)                  # picks per-row
        >>> y.data
        array([[12., 10.],
            [21., 21.]], dtype=float32)
        """
        out_data = self.backend.take_along_axis(self.data, index.data, axis=dim)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            grad = self.backend.zeros_like(self.data)
            self.backend.put_along_axis(grad, index.data, out.grad, axis=dim)
            Tensor._accumulate_grad(self, grad)

        out._backward = _backward

        return out

    def exp(self) -> "Tensor":
        """
        Element-wise exponential function.

        Computes :math:``y_i = e^{x_i}`` for each element in the tensor.

        Returns
        -------
        Tensor
            A new tensor with the exponential of each element.

        Notes
        -----
        - Equivalent to ``numpy.exp`` or ``torch.exp``.
        - **Gradient:** The derivative of ``exp(x)`` is ``exp(x)`` itself, so in
        the backward pass, the gradient is simply ``out * out.grad``.

        Examples
        --------
        >>> x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        >>> y = x.exp()
        >>> y.data
        array([1.0, 2.7182817, 7.389056], dtype=float32)
        >>> y.backward(Tensor([1.0, 1.0, 1.0]))
        >>> x.grad
        array([1.0, 2.7182817, 7.389056], dtype=float32)
        """
        out_data = self.backend.exp(self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, out.data * out.grad)
        out._backward = _backward

        return out

    def log(self) -> "Tensor":
        """
        Element-wise natural logarithm.

        Computes :math:``y_i = \log(x_i)`` for each element of the tensor.

        Returns
        -------
        Tensor
            A new tensor with the natural logarithm of each element.

        Notes
        -----
        - Equivalent to ``numpy.log`` or ``torch.log``.
        - **Gradient:** ``d(log(x)) / dx = 1 / x``.
        - Input values must be positive to avoid NaNs or ``-inf``.

        Examples
        --------
        >>> x = Tensor([1.0, 2.71828], requires_grad=True)
        >>> y = x.log()
        >>> y.data
        array([0.0, 1.0], dtype=float32)
        """
        out_data = self.backend.log(self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, out.grad / self.data)
        out._backward = _backward

        return out

    def tanh(self) -> "Tensor":
        """
        Element-wise hyperbolic tangent.

        Computes :math:``y_i = \\tanh(x_i)`` for each element of the tensor.

        Returns
        -------
        Tensor
            A new tensor with the hyperbolic tangent of each element.

        Notes
        -----
        - Equivalent to ``numpy.tanh`` or ``torch.tanh``.
        - **Gradient:** ``d(tanh(x)) / dx = 1 - tanh(x)^2``.
        - Commonly used as an activation function in neural networks.

        Examples
        --------
        >>> x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> y = x.tanh()
        >>> y.data
        array([-0.7615942, 0.0, 0.7615942], dtype=float32)
        """
        out_data = self.backend.tanh(self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, (1 - out.data**2) * out.grad)
        out._backward = _backward

        return out

    def sigmoid(self) -> "Tensor":
        """
        Element-wise logistic sigmoid function.

        Computes :math:``y_i = \\frac{1}{1 + e^{-x_i}}`` for each element of the tensor.

        Returns
        -------
        Tensor
            A new tensor containing the sigmoid of each element.

        Notes
        -----
        - Equivalent to ``torch.sigmoid`` or ``scipy.special.expit``.
        - **Gradient:** ``d(sigmoid(x)) / dx = sigmoid(x) * (1 - sigmoid(x))``.
        - Commonly used as an activation function in neural networks.

        Examples
        --------
        >>> x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> y = x.sigmoid()
        >>> y.data
        array([0.26894143, 0.5, 0.7310586], dtype=float32)
        """
        out_data = 1 / (1 + self.backend.exp(-self.data))
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, (out.data - out.data**2) * out.grad)
        out._backward = _backward

        return out

    def relu(self) -> "Tensor":
        """
        Element-wise Rectified Linear Unit (ReLU) activation.

        Computes :math:``y_i = \\max(0, x_i)`` for each element of the tensor.

        Returns
        -------
        Tensor
            A new tensor where all negative values are replaced by zero.

        Notes
        -----
        - Equivalent to ``torch.relu`` or ``numpy.maximum(0, x)``.
        - **Gradient:** ``d(ReLU(x)) / dx = 1`` if ``x > 0``, else ``0``.
        - Commonly used as a non-linear activation function in deep networks.

        Examples
        --------
        >>> x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
        >>> y = x.relu()
        >>> y.data
        array([0.0, 0.0, 2.0], dtype=float32)
        """
        out_data = self.backend.maximum(0, self.data)
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            Tensor._accumulate_grad(self, (self.data > 0).astype(self.data.dtype) * out.grad)
        out._backward = _backward

        return out

    def gelu(self) -> "Tensor":
        """
        Element-wise Gaussian Error Linear Unit (GELU) activation (approximate form).

        Computes the approximate GELU function:

        .. math::
            \\text{GELU}(x) \\approx 0.5x\\left[1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}}(x + 0.044715x^3)\\right)\\right]

        Returns
        -------
        Tensor
            A new tensor containing the GELU activation of each element.

        Notes
        -----
        - This is the **approximate** GELU variant used in BERT and GPT models.
        - Equivalent to ``torch.nn.functional.gelu(x, approximate="tanh")``.
        - Smoothly blends between linear and zero activation regions, unlike ReLU.

        Examples
        --------
        >>> x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> y = x.gelu()
        >>> y.data
        array([-0.1588, 0.0, 0.8413], dtype=float32)
        """
        c = (2 / self.backend.pi) ** 0.5
        return 0.5 * self * (1 + ((self + 0.044715 * self ** 3) * c).tanh())
    
    def logsumexp(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
    ) -> "Tensor":
        """
        Numerically stable log-sum-exp reduction.

        Computes:

        .. math::
            \\text{logsumexp}(x) = \\log\\sum_i e^{x_i}

        but in a numerically stable way by subtracting the maximum value along
        the specified dimension before exponentiation.

        Parameters
        ----------
        dim : int or tuple of ints, optional
            The dimension(s) along which to reduce. If ``None``, all elements are reduced.
        keepdim : bool, default=False
            If True, retains reduced dimensions with length 1.

        Returns
        -------
        Tensor
            A tensor containing the log-sum-exp values along the specified dimension(s).

        Notes
        -----
        - This implementation avoids overflow by using ``x - max(x)`` shifting.
        - Equivalent to ``torch.logsumexp`` or ``scipy.special.logsumexp``.
        - Commonly used in softmax and cross-entropy calculations.

        Examples
        --------
        >>> x = Tensor([[1.0, 2.0, 3.0]])
        >>> x.logsumexp(dim=1)
        tensor([3.4076], dtype=float32)
        """
        max_val = self.max(dim=dim, keepdim=True)

        shifted = self - max_val
        exp_shifted = shifted.exp()

        sum_exp = exp_shifted.sum(dim=dim, keepdim=keepdim)
        log_sum_exp = sum_exp.log()

        if keepdim:
            return log_sum_exp + max_val
        else:
            return log_sum_exp + max_val.squeeze(dim)

    def log_softmax(
        self,
        dim: Optional[int] = None,
    ) -> "Tensor":
        """
        Computes the log of the softmax function along the given dimension.

        Uses the numerically stable formulation:
        log_softmax(x) = x - logsumexp(x).

        Parameters
        ----------
        dim : int, optional
            Dimension along which to apply softmax.

        Returns
        -------
        Tensor
            Tensor of log-softmax values.
        """
        return self - self.logsumexp(dim=dim, keepdim=True)
    
    def pad2d(
        self,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
    ) -> "Tensor":
        """
        Applies zero-padding to the spatial dimensions (H, W) of a 4D tensor.

        Parameters
        ----------
        padding : int or tuple
            Padding configuration:
            - ``int p`` → pad all sides by ``p``.
            - ``(pH, pW)`` → pad height and width symmetrically.
            - ``(t, b, l, r)`` → pad top, bottom, left, and right separately.

        Returns
        -------
        Tensor
            Padded tensor of shape ``(N, C, H + t + b, W + l + r)``.

        Notes
        -----
        - Padding is applied using zeros (constant mode).
        - The backward pass removes the padded regions from the gradient.
        - Equivalent to ``torch.nn.functional.pad(x, pad=(l, r, t, b))`` with ``mode="constant"``.
        """
        if isinstance(padding, int):
            t = b = l = r = padding
        elif len(padding) == 2:
            t = b = int(padding[0]); l = r = int(padding[1])
        else:
            t, b, l, r = map(int, padding)

        out_data = self.backend.pad(self.data, ((0,0),(0,0),(t,b),(l,r)), mode="constant")
        out = Tensor(out_data, _prev=(self,), requires_grad=self.requires_grad)

        def _backward():
            g = out.grad[:, :, t:t+self.shape[2], l:l+self.shape[3]]
            Tensor._accumulate_grad(self, g)

        out._backward = _backward
        return out

    def backward(
        self,
        gradient: Optional[Any] = None,
    ) -> None:
        """
        Performs backpropagation through the computation graph, computing gradients
        for all tensors that have ``requires_grad=True``.

        This method initiates a reverse traversal of the graph (topological order)
        and accumulates gradients for all dependencies using each tensor’s
        registered ``_backward`` function.

        Parameters
        ----------
        gradient : array-like, optional
            Gradient of the output with respect to itself.
            - If ``gradient`` is None, uses ``ones_like(self.data)`` (this allows calling ``backward()`` on non-scalar tensors, unlike PyTorch).

        Raises
        ------
        RuntimeError
            If the tensor was created with ``requires_grad=False``.

        Notes
        -----
        - The method constructs a topological ordering of the computation graph
        before performing backpropagation to ensure correct gradient propagation.
        - Gradients are accumulated in the ``.grad`` attribute of each tensor.
        - This is the main entry point for autograd in the framework.

        Examples
        --------
        >>> x = Tensor([2.0, 3.0], requires_grad=True)
        >>> y = (x * x).sum()
        >>> y.backward()
        >>> x.grad
        tensor([4.0, 6.0], dtype=float32)
        """
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require gradient")
        if gradient is None:
            self.grad = self.backend.ones_like(self.data)
        else:
            self.grad = self.backend.array(gradient, dtype=self.data.dtype)

        visited = set()
        topo = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            if t.requires_grad:
                t._backward()

    def zero_grad(self) -> None:
        """
        Resets the gradient of this tensor to zero in place.

        Notes
        -----
        - This is typically called before each optimization step to prevent
        gradient accumulation across iterations.
        - Equivalent to ``torch.Tensor.grad.zero_()`` in PyTorch.
        """
        if self.requires_grad:
          self.grad = self.backend.zeros_like(self.data)

    def __repr__(self) -> str:
        """
        Returns a readable string representation of the tensor.

        The representation includes the tensor's data, dtype, device, and
        whether gradients are tracked. For CuPy-backed tensors, the device
        is shown as ``'cuda'``; otherwise, ``'cpu'``.

        Returns
        -------
        str
            A formatted string describing the tensor.

        Examples
        --------
        >>> x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        >>> print(x)
        tensor([[1., 2.],
                [3., 4.]], dtype=float32, requires_grad=True, device='cpu')
        """
        data_str = self.backend.array2string(self.data, separator=', ', prefix='tensor(')
        details = [f"dtype={self.data.dtype}, requires_grad={self.requires_grad}"]

        dev = "cuda" if (_HAS_CUPY and self.backend is cp) else "cpu"
        details.append(f"device='{dev}'")

        return f"tensor({data_str}, {', '.join(details)})"

    def to(
        self,
        device: str,
    ) -> "Tensor":
        """
        Moves the tensor to the specified device (CPU or CUDA).

        This method converts both the tensor data and its gradient (if present)
        between NumPy and CuPy backends as needed. The operation is performed
        in-place and returns the same tensor for convenience.

        Parameters
        ----------
        device : str
            Target device to move the tensor to. Must be one of:
            - ``"cpu"`` — use NumPy backend.
            - ``"cuda"`` — use CuPy backend (requires CuPy to be installed).

        Returns
        -------
        Tensor
            The same tensor instance, now residing on the specified device.

        Raises
        ------
        RuntimeError
            If ``"cuda"`` is requested but CuPy is not installed or available.

        Examples
        --------
        >>> x = Tensor([[1, 2], [3, 4]], device="cpu")
        >>> x = x.to("cuda")   # moves to GPU if available
        >>> x.device
        'cuda'
        >>> x = x.to("cpu")    # back to CPU
        >>> x.device
        'cpu'
        """
        dev = _normalize_device(device)
        if dev == "cpu" and _HAS_CUPY and self.backend is cp:
            self.data = cp.asnumpy(self.data)
            self.grad = cp.asnumpy(self.grad) if self.grad is not None else None
            self.backend = np
        elif dev == "cuda" and self.backend is not (cp if _HAS_CUPY else None):
            if not _HAS_CUPY:
                raise RuntimeError("CUDA requested but CuPy is not installed/available.")
            self.data = cp.asarray(self.data)
            self.grad = cp.asarray(self.grad) if self.grad is not None else None
            self.backend = cp
        return self

    def xp(self) -> Any:
        """Return the current array backend (NumPy or CuPy)."""
        return self.backend
    
    @staticmethod
    def _cat(
        tensors: Sequence["Tensor"],
        dim: int = 0,
    ) -> "Tensor":
        """
        Concatenate tensors along a given dimension (NumPy/CuPy semantics).

        All tensors must:
        - use the same backend (NumPy or CuPy),
        - have identical shapes on all axes **except** the chosen ``dim``.

        The backward pass splits the upstream gradient along ``dim`` and
        accumulates each slice into the corresponding input tensor.

        Parameters
        ----------
        tensors : sequence of Tensor
            Non-empty list/tuple of tensors to concatenate.
        dim : int, default=0
            Concatenation dimension. Negative values are supported
            (interpreted modulo the rank of the input tensors).

        Returns
        -------
        Tensor
            Concatenated tensor. ``requires_grad`` is True if any input requires grad.
        """
        backend = tensors[0].backend
        data = backend.concatenate([t.data for t in tensors], axis=dim)
        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(data, _prev=tuple(tensors), requires_grad=requires_grad)

        def _backward():
            sizes = [t.shape[dim] for t in tensors]
            start = 0
            for t, sz in zip(tensors, sizes):
                slc = [slice(None)] * out.grad.ndim
                slc[dim] = slice(start, start + sz)
                Tensor._accumulate_grad(t, out.grad[tuple(slc)])
                start += sz

        out._backward = _backward
        return out

    @staticmethod
    def _im2col(
        x: "Tensor",
        kH: int, kW: int,
        sH: int, sW: int,
        dH: int, dW: int,
        Hout: int, Wout: int,
        pH: int, pW: int,
    ) -> "Tensor":
        """
        Convert a 4D image batch into a 2D "column" matrix (im2col) for convolution.

        This helper rearranges sliding local blocks from an input tensor into a
        matrix suitable for expressing a 2D convolution as a matrix multiplication.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(N, C, H, W)``.
        kH, kW : int
            Kernel (filter) height and width.
        sH, sW : int
            Stride along height and width.
        dH, dW : int
            Dilation along height and width.
        Hout, Wout : int
            Output spatial size that the convolution will produce.
            These are typically computed as:
            ``Hout = floor((H + 2*pH - dH*(kH-1) - 1)/sH) + 1`` and similarly for Wout.
        pH, pW : int
            Zero-padding applied symmetrically to the input along height and width.

        Returns
        -------
        Tensor
            Tensor ``Xcols`` of shape ``(N, C*kH*kW, Hout*Wout)``.
            Each column corresponds to one spatial output position (flattened over
            ``Hout*Wout``), and each column contains the flattened receptive field
            across channels and kernel elements.

        Notes
        -----
        - This function uses :meth:``Tensor.pad2d``, slicing, reshape, and concatenate,
        so gradients flow back to ``x`` through the autograd graph.
        - The layout matches common im2col conventions used to implement convolution
        efficiently via GEMM:
            - reshape to ``(N, C*kH*kW, Hout*Wout)``
            - later, a weight matrix of shape ``(Cout, C*kH*kW)`` can multiply it.

        Examples
        --------
        >>> x = Tensor.randn(2, 3, 5, 5, requires_grad=True)  # (N=2,C=3,H=5,W=5)
        >>> Xcols = Tensor._im2col(x, kH=3, kW=3, sH=1, sW=1, dH=1, dW=1,
        ...                        Hout=3, Wout=3, pH=0, pW=0)
        >>> Xcols.shape
        (2, 27, 9)
        """
        x_pad = x.pad2d((pH, pH, pW, pW))

        cols = []
        for ky in range(kH):
            y0 = ky * dH
            y1 = y0 + sH * Hout
            for kx in range(kW):
                x0 = kx * dW
                x1 = x0 + sW * Wout
                patch = x_pad[:, :, y0:y1:sH, x0:x1:sW]
                cols.append(patch.reshape(x.shape[0], x.shape[1], -1))
        Xcols = Tensor._cat(cols, dim=1)
        N, C = x.shape[0], x.shape[1]
        K = kH * kW
        HW = Hout * Wout

        Xcols = Xcols.reshape(N, K, C, HW)        # (N, kH*kW, C, HW)
        Xcols = Xcols.permute(0, 2, 1, 3)         # (N, C, kH*kW, HW)
        Xcols = Xcols.reshape(N, C * K, HW)       # (N, C*kH*kW, HW)
        return Xcols

    @staticmethod
    def _expand_like(
        x: Any,
        target_shape: Tuple[int, ...],
        dims_reduced: Union[int, Tuple[int, ...]],
    ) -> Any:
        """
        Expand an array to ``target_shape`` by re-inserting reduced dimensions and broadcasting.

        Parameters
        ----------
        x : numpy.ndarray or cupy.ndarray
            Input array (backend array, not a Tensor).
        target_shape : tuple[int, ...]
            Desired output shape.
        dims_reduced : int or tuple[int, ...]
            Axis/axes that were reduced and should be re-inserted as singleton dims.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            Broadcasted view/array with shape ``target_shape``.
        """
        if isinstance(dims_reduced, int):
            dims_reduced = (dims_reduced,)

        backend = cp if (_HAS_CUPY and isinstance(x, cp.ndarray)) else np

        for dim in sorted(dims_reduced):
            x = backend.expand_dims(x, axis=dim)

        return backend.broadcast_to(x, target_shape)

    @staticmethod
    def _unbroadcast(
        x: Any,
        target_shape: Tuple[int, ...],
    ) -> Any:
        """
        Reduce a broadcasted gradient ``x`` back to ``target_shape`` by summing over broadcasted axes.

        Parameters
        ----------
        x : numpy.ndarray or cupy.ndarray
            Gradient array with broadcasted shape.
        target_shape : tuple[int, ...]
            Original (pre-broadcast) shape to reduce to.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            Reduced gradient with shape ``target_shape``.
        """
        while x.ndim > len(target_shape):
            x = x.sum(axis=0)
        for i, (g, t) in enumerate(zip(x.shape, target_shape)):
            if g != t:
                x = x.sum(axis=i, keepdims=True)

        return x.reshape(target_shape)

    @staticmethod
    def _ensure_tensor(
        x: Union["Tensor", Any], 
        backend: Any,
    ) -> "Tensor":
        """
        Ensure that ``x`` is a :class:``Tensor`` on the specified backend.

        If ``x`` is already a ``Tensor``, it is returned unchanged. Otherwise,
        ``x`` is converted to a new tensor and placed on the device implied
        by ``backend`` (NumPy → CPU, CuPy → CUDA).

        Notes
        -----
        - This helper is used to support mixed operations between tensors and
        Python scalars or array-like objects (e.g. ``Tensor + 3``).
        """
        if isinstance(x, Tensor):
            return x
        return Tensor(x, device="cuda" if _HAS_CUPY and backend is cp else "cpu")

    @staticmethod
    def _accumulate_grad(
        tensor: "Tensor",
        grad: Any,
    ) -> None:
        """
        Accumulate a gradient contribution into ``tensor.grad``.

        If ``tensor.requires_grad`` is True, the provided gradient is added
        to the tensor’s existing gradient buffer. If no gradient buffer
        exists yet, it is initialized with ``grad``.

        Notes
        -----
        - Gradients are accumulated (not overwritten) because a tensor may
        contribute to the output through multiple paths in the computation graph.
        - This mirrors PyTorch’s gradient accumulation semantics for leaf tensors.
        - No operation is performed if ``tensor.requires_grad`` is False.
        """
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = grad
            else:
                tensor.grad += grad

    @staticmethod
    def zeros(
        *shape: int,
        requires_grad: bool = False,
        device: Optional[str] = "cpu",
    ) -> "Tensor":
        """
        Create a tensor filled with zeros.

        Parameters
        ----------
        *shape : int
            Shape of the output tensor.
        requires_grad : bool, default=False
            If True (and global grad mode is enabled), operations on the tensor
            will be tracked for automatic differentiation.
        device : str or None, default="cpu"
            Target device for the tensor (``"cpu"`` or ``"cuda"``).

        Returns
        -------
        Tensor
            A float32 tensor of zeros with the specified shape and device.

        Notes
        -----
        - Equivalent to ``torch.zeros`` (restricted to float32).
        - The backend (NumPy or CuPy) is selected based on ``device``.
        """
        dev = _normalize_device(device) or "cpu"
        if dev == "cuda":
            if not _HAS_CUPY:
                raise RuntimeError("CUDA requested but CuPy is not installed/available.")
            xp = cp
        else:
            xp = np
        data = xp.zeros(shape, dtype=xp.float32)
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def randn(
        *shape: int,
        requires_grad: bool = False,
        scale: float = 1.0,
        device: Optional[str] = "cpu",
    ) -> "Tensor":
        """
        Create a tensor with values sampled from a normal distribution.

        Samples i.i.d. values from ``N(0, 1)`` and scales them by ``scale``,
        resulting in a distribution ``N(0, scale^2)``.

        Parameters
        ----------
        *shape : int
            Shape of the output tensor.
        requires_grad : bool, default=False
            If True (and global grad mode is enabled), operations on the tensor
            will be tracked for automatic differentiation.
        scale : float, default=1.0
            Multiplicative scale applied to the sampled values.
        device : str or None, default="cpu"
            Target device for the tensor (``"cpu"`` or ``"cuda"``).

        Returns
        -------
        Tensor
            A float32 tensor with normally distributed values.

        Notes
        -----
        - Equivalent to ``torch.randn`` with an explicit scaling factor.
        - Commonly used for parameter initialization.
        """
        dev = _normalize_device(device) or "cpu"
        if dev == "cuda":
            if not _HAS_CUPY:
                raise RuntimeError("CUDA requested but CuPy is not installed/available.")
            xp = cp
        else:
            xp = np
        data = scale * xp.random.randn(*shape).astype(xp.float32)
        return Tensor(data, requires_grad=requires_grad)
    
    @staticmethod
    def randint(
        low: int,
        high: int,
        size: Union[int, Tuple[int, ...]],
        requires_grad: bool = False,
        device: Optional[str] = "cpu",
    ) -> "Tensor":
        """
        Create a tensor with values sampled from a normal distribution.

        Samples i.i.d. values from ``N(0, 1)`` and scales them by ``scale``,
        resulting in a distribution ``N(0, scale^2)``.

        Parameters
        ----------
        *shape : int
            Shape of the output tensor.
        requires_grad : bool, default=False
            If True (and global grad mode is enabled), operations on the tensor
            will be tracked for automatic differentiation.
        scale : float, default=1.0
            Multiplicative scale applied to the sampled values.
        device : str or None, default="cpu"
            Target device for the tensor (``"cpu"`` or ``"cuda"``).

        Returns
        -------
        Tensor
            A float32 tensor with normally distributed values.

        Notes
        -----
        - Equivalent to ``torch.randn`` with an explicit scaling factor.
        - Commonly used for parameter initialization.
        """
        dev = _normalize_device(device) or "cpu"
        if dev == "cuda":
            if not _HAS_CUPY:
                raise RuntimeError("CUDA requested but CuPy is not installed/available.")
            xp = cp
        else:
            xp = np
        data = scale * xp.random.randint(*shape).astype(xp.float32)
        return Tensor(data, requires_grad=requires_grad)