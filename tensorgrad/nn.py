from typing import Any, Dict, List, Sequence, Tuple, Union
from tensorgrad.tensor import Tensor

class Module:
    """
    Base class for all neural network modules.

    Modules can contain:
    - submodules (instances of :class:`Module`)
    - parameters (instances of :class:`Tensor`)

    Submodules and parameters assigned as attributes are registered automatically
    via :meth:`__setattr__`. The public API mirrors a minimal subset of PyTorch's
    ``torch.nn.Module``.
    """
    def __init__(self) -> None:
        """
        Initialize an empty module.

        Attributes
        ----------
        _modules : dict[str, Module]
            Registered child modules.
        _parameters : dict[str, Tensor]
            Registered parameters.
        training : bool
            If True, the module is in training mode (affects modules like Dropout/BatchNorm).
        """
        self._modules = {}
        self._parameters = {}
        self.training = True

    def parameters(self) -> List[Tensor]:
        """
        Return a flat list of all parameters in this module and its submodules.

        Returns
        -------
        list[Tensor]
            Parameters in a deterministic traversal order: local parameters first,
            then parameters of children in insertion order.
        """
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self) -> None:
        """
        Set gradients of all parameters to zero.

        Notes
        -----
        This calls :meth:`Tensor.zero_grad` on each parameter.
        """
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        """
        Set training mode for this module and all submodules.

        Parameters
        ----------
        mode : bool, default=True
            If True, enables training mode. If False, enables evaluation mode.

        Returns
        -------
        Module
            ``self`` (to allow chaining).
        """
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """
        Set evaluation mode for this module and all submodules.

        Returns
        -------
        Module
            ``self``.
        """
        return self.train(False)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Register submodules and parameters assigned as attributes.

        Notes
        -----
        - Assigning a :class:`Module` registers it in ``self._modules``.
        - Assigning a :class:`Tensor` registers it in ``self._parameters``.
        - Everything is still set as a normal attribute via ``super().__setattr__``.
        """
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def __repr__(self):
        modstr = self._modules.items()
        lines = [f"{self.__class__.__name__}("]
        for name, module in modstr:
            mod_repr = repr(module)
            mod_repr = "\n    ".join(mod_repr.splitlines())
            lines.append(f"  ({name}): {mod_repr}")
        lines.append(")")
        return "\n".join(lines)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return a state dictionary of parameter values.

        Returns
        -------
        dict
            Maps parameter names to **copies** of their underlying arrays.
            Submodule parameters use dotted keys (e.g. ``"layer1.weight"``).

        Notes
        -----
        - This serializes only parameter *data* (not gradients).
        - Buffers (e.g. BatchNorm running stats) are not included unless you
          register them as Tensors and treat them as parameters/buffers explicitly.
        """
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data.copy()

        for name, module in self._modules.items():
            sub_state = module.state_dict()
            for sub_name, value in sub_state.items():
                state[f"{name}.{sub_name}"] = value

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load parameter values from a state dictionary.

        Parameters
        ----------
        state_dict : dict
            Dictionary produced by :meth:`state_dict`.

        Raises
        ------
        KeyError
            If a required parameter key is missing.

        Notes
        -----
        This loads parameter data in-place: ``param.data[:] = state_dict[name]``.
        """
        for name, param in self._parameters.items():
            if name in state_dict:
                param.data[:] = state_dict[name]
            else:
                raise KeyError(f"{name} not found in state_dict")

        for name, module in self._modules.items():
            sub_state = {
                k[len(name) + 1:]: v
                for k, v in state_dict.items()
                if k.startswith(f"{name}.")
            }
            module.load_state_dict(sub_state)

    def to(self, device: str) -> "Module":
        """
        Move all parameters (and submodules) to the specified device.

        Parameters
        ----------
        device : str
            Target device, e.g. ``"cpu"`` or ``"cuda"`` (and variants like ``"cuda:0"``).

        Returns
        -------
        Module
            ``self``.
        """
        for name, param in self._parameters.items():
            param.to(device)
        for module in self._modules.values():
            module.to(device)
        return self
    
class Linear(Module):
    """
    Fully-connected linear layer.

    Computes ``y = x @ W^T + b``.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, default=True
        If True, includes a learnable bias.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        gain = (2. / in_features) ** 0.5
        self.weight = Tensor.randn(out_features, in_features, requires_grad=True, scale=gain)
        self.bias = Tensor.zeros(out_features, requires_grad=True) if bias else None

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, bias={self.bias is not None})"

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(..., in_features)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(..., out_features)``.
        """
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
    
class ReLU(Module):
    """Element-wise ReLU activation: ``max(0, x)``."""
    def __repr__(self):
      return f"{self.__class__.__name__}()"

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
class Tanh(Module):
    """Element-wise hyperbolic tangent activation."""
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
    
class Sequential(Module):
    """
    A container module that applies submodules in sequence.

    Parameters
    ----------
    *modules : Module
        Modules applied in the given order.
    """
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self._modules_list = []

        for idx, module in enumerate(modules):
            assert isinstance(module, Module), f"All elements must be Module instances, got {type(module)}"
            self._modules_list.append(module)
            self._modules[str(idx)] = module

    def forward(self, x: Tensor) -> Tensor:
        """Apply each module to the output of the previous one."""
        for module in self._modules_list:
            x = module(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        """Return the module at position ``idx``."""
        return self._modules_list[idx]
    
class MSELoss(Module):
    """
    Mean-squared error loss.

    Parameters
    ----------
    reduction : {'mean', 'sum'}, default='mean'
        Reduction applied to the per-element squared error.
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction})"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Predicted values.
        target : Tensor
            Target values (same shape as ``input``).

        Returns
        -------
        Tensor
            Scalar loss if reduced, otherwise a tensor of per-element losses.
        """
        diff = input - target
        loss = diff * diff

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
        
class CrossEntropyLoss(Module):
    """
    Cross-entropy loss for multi-class classification.

    Expects unnormalized logits and integer class targets.

    Parameters
    ----------
    reduction : {'mean', 'sum'}, default='mean'
        Reduction applied over the batch.
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction})"

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Parameters
        ----------
        logits : Tensor
            Logits of shape ``(B, C)`` where ``C`` is number of classes.
        targets : Tensor
            Integer class indices of shape ``(B,)`` (values in ``[0, C)``).

        Returns
        -------
        Tensor
            Reduced scalar loss (mean or sum over batch).
        """
        log_probs = logits.log_softmax(dim=1)                         # shape: (B, C)
        targets = targets.unsqueeze(1)                                # shape: (B, 1)
        picked_log_probs = log_probs.gather(dim=1, index = targets)   # shape: (B, 1)
        loss = -picked_log_probs.squeeze(1)                           # shape: (B,)

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
        
class Dropout(Module):
    """
    Dropout regularization.

    During training, randomly zeros elements with probability ``p`` and scales
    the remaining elements by ``1/(1-p)`` (inverted dropout).

    Parameters
    ----------
    p : float, default=0.5
        Probability of an element to be zeroed.
    """
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout if in training mode.

        Notes
        -----
        This implementation generates a random mask on the same backend as ``x``.
        """
        if self.training:
            backend = x.backend
            mask = (backend.random.rand(*x.shape) > self.p).astype(x.data.dtype)
            return x * Tensor(mask, requires_grad=False) / (1 - self.p)
        else:
            return x
        
class BatchNorm1d(Module):
    """
    Batch Normalization over a channel dimension.

    Normalizes input using batch statistics during training and running
    statistics during evaluation.

    Parameters
    ----------
    num_features : int
        Number of features/channels in dimension 1.
    eps : float, default=1e-5
        Small constant for numerical stability.
    momentum : float, default=0.1
        Momentum for running mean/variance updates.
    affine : bool, default=True
        If True, includes learnable scale (weight) and shift (bias).
    track_running_stats : bool, default=True
        If True, maintains running mean/variance for evaluation mode.
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Tensor.ones(num_features, requires_grad=True)
            self.bias = Tensor.zeros(num_features, requires_grad=True)
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = Tensor.zeros(num_features)
            self.running_var = Tensor.ones(num_features)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input of shape ``(N, C, ...)`` where ``C == num_features``.

        Returns
        -------
        Tensor
            Batch-normalized tensor with the same shape as input.
        """
        if self.training:
            reduce_dims = tuple(i for i in range(x.ndim) if i != 1)
            mean = x.mean(dim=reduce_dims, keepdim=True)
            var = x.var(dim=reduce_dims, keepdim=True, unbiased=False)

            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.reshape((1, -1) + (1,) * (x.ndim - 2))
            var = self.running_var.reshape((1, -1) + (1,) * (x.ndim - 2))

        x_hat = (x - mean) / ((var + self.eps) ** 0.5)

        if self.affine:
            w = self.weight.reshape((1, -1) + (1,) * (x.ndim - 2))
            b = self.bias.reshape((1, -1) + (1,) * (x.ndim - 2))
            return x_hat * w + b
        else:
            return x_hat
        
class LayerNorm(Module):
    """
    Layer Normalization over the last ``len(normalized_shape)`` dimensions.

    Parameters
    ----------
    normalized_shape : int or sequence of int
        Input shape over which to normalize (typically the feature dimension(s)).
    eps : float, default=1e-5
        Numerical stability constant.
    elementwise_affine : bool, default=True
        If True, includes learnable scale (weight) and optionally bias.
    bias : bool, default=True
        If True and ``elementwise_affine`` is True, includes a learnable bias.
    """
    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        self.has_bias = bias

        if elementwise_affine:
            self.weight = Tensor.ones(*self.normalized_shape, requires_grad=True)
            self.bias = Tensor.zeros(*self.normalized_shape, requires_grad=True) if bias else None
        else:
            self.weight = None
            self.bias = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, bias={self.has_bias})"

    def forward(self, x: Tensor) -> Tensor:
        """Normalize input and apply optional affine transform."""
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_hat = (x - mean) / ((var + self.eps) ** 0.5)

        if self.elementwise_affine:
            x_hat = x_hat * self.weight
            if self.bias is not None:
                x_hat = x_hat + self.bias
        return x_hat
    
class Embedding(Module):
    """
    Lookup table that maps integer indices to dense vectors.

    Parameters
    ----------
    num_embeddings : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of each embedding vector.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor.randn(num_embeddings, embedding_dim, requires_grad=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"

    def forward(self, indices: Tensor) -> Tensor:
        """
        Parameters
        ----------
        indices : Tensor
            Integer indices tensor of arbitrary shape. Values must be in
            ``[0, num_embeddings)``.

        Returns
        -------
        Tensor
            Embedded tensor of shape ``indices.shape + (embedding_dim,)``.

        Notes
        -----
        Gradients are accumulated into ``weight`` by scattering (summing) the
        upstream gradients for repeated indices.
        """
        backend = self.weight.backend
        embedded_data = self.weight.data[indices.data.astype(backend.int64)]
        out = Tensor(embedded_data, _prev=(self.weight,), requires_grad=self.weight.requires_grad)

        def _backward():
            grad_output = out.grad
            backend = self.weight.backend
            flat_indices = indices.data.ravel().astype(backend.int64)
            flat_grads = grad_output.reshape(-1, self.embedding_dim)
            grad_weight = backend.zeros_like(self.weight.data)
            backend.add.at(grad_weight, flat_indices, flat_grads)

            Tensor._accumulate_grad(self.weight, grad_weight)

        out._backward = _backward
        return out

class Conv2d(Module):
    """
    2D convolution layer (NCHW).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int]
        Kernel height and width.
    stride : int or tuple[int, int], default=1
        Stride along height and width.
    padding : int or tuple[int, int], default=0
        Zero-padding along height and width.
    dilation : int or tuple[int, int], default=1
        Dilation along height and width.
    bias : bool, default=True
        If True, includes a learnable bias.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        kH, kW = self.kernel_size
        fan_in = in_channels * kH * kW
        scale = (2.0 / fan_in) ** 0.5
        self.weight = Tensor.randn(out_channels, in_channels, kH, kW, requires_grad=True, scale=scale)
        self.bias = Tensor.zeros(out_channels, requires_grad=True) if bias else None

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, dilation={self.dilation}, "
                f"bias={self.bias is not None})")

    @staticmethod
    def _out_hw(
        H: int, W: int,
        kH: int, kW: int,
        pH: int, pW: int,
        sH: int, sW: int,
        dH: int, dW: int,
    ) -> Tuple[int, int]:
        """
        Compute output spatial size for a 2D convolution.

        Returns
        -------
        (Hout, Wout) : tuple[int, int]
            Output height and width.
        """
        eff_kH = (kH - 1) * dH + 1
        eff_kW = (kW - 1) * dW + 1
        Hout = (H + 2 * pH - eff_kH) // sH + 1
        Wout = (W + 2 * pW - eff_kW) // sW + 1
        return int(Hout), int(Wout)
    
    def forward(self, x: Tensor):
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(N, C_in, H, W)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(N, C_out, H_out, W_out)``.
        """
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        N, _, H, W = x.shape
        Hout, Wout = self._out_hw(H, W, kH, kW, pH, pW, sH, sW, dH, dW)

        Xcols = Tensor._im2col(x, kH, kW, sH, sW, dH, dW, Hout, Wout, pH, pW)

        Wcol = self.weight.reshape(self.out_channels, -1)
        Ymat = (Wcol @ Xcols)        
        Y = Ymat.reshape(N, self.out_channels, Hout, Wout)

        if self.bias is not None:
            Y = Y + self.bias.reshape(1, -1, 1, 1)

        return Y