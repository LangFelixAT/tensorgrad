from typing import Any, Dict, Iterable, Tuple
from src.tensor import Tensor

class Optimizer:
    """
    Base class for all optimizers.

    An optimizer updates a collection of parameters (tensors) in-place based on
    their gradients. Subclasses must implement :meth:`step` and the state
    serialization helpers.

    Parameters
    ----------
    params : Iterable[Tensor]
        Iterable of parameters to optimize. Parameters are stored as a list and
        iterated in the given order.

    Notes
    -----
    - This class mirrors the high-level structure of ``torch.optim.Optimizer``:
      it maintains a per-parameter ``state`` and supports ``state_dict()``
      serialization.
    - The optimizer assumes parameters expose at least:
      ``.data`` (ndarray), ``.grad`` (ndarray or None), ``.requires_grad`` (bool),
      ``.zero_grad()`` and ``.backend`` (numpy/cupy module).
    """
    def __init__(self, params: Iterable["Tensor"]) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        """
        Reset gradients of all parameters to zero.

        Notes
        -----
        - Only parameters with ``requires_grad=True`` are affected.
        - Equivalent to calling ``p.zero_grad()`` for each parameter.
        """
        for p in self.params:
            if p.requires_grad:
                p.zero_grad()

    def step(self) -> None:
        """
        Perform a single optimization step.

        Subclasses must implement this method to update each parameter using its
        gradient and any optimizer-specific state.
        """
        raise NotImplementedError
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return the optimizer state as a Python dictionary.

        Returns
        -------
        dict
            Dictionary containing:
            - ``"hyperparams"``: optimizer hyperparameters (subclass-defined)
            - ``"state"``: per-parameter state entries aligned with ``self.params``

        Notes
        -----
        - Per-parameter state is stored by parameter identity in ``self.state``.
          This method serializes it into a list matching parameter order.
        """
        return {
            "hyperparams": self._get_hyperparams(),
            "state": [
                self._serialize_param_state(p) if p in self.state else None
                for p in self.params
            ],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load optimizer state from a dictionary produced by :meth:`state_dict`.

        Parameters
        ----------
        state_dict : dict
            Optimizer state dictionary.

        Notes
        -----
        - Assumes ``self.params`` correspond to the same model parameters as when
          the state was saved (order matters).
        - Hyperparameters are restored first, then per-parameter buffers.
        """
        self._set_hyperparams(state_dict["hyperparams"])
        self.state = {}
        for p, s in zip(self.params, state_dict["state"]):
            if s is not None:
                self._deserialize_param_state(p, s)

    def _get_hyperparams(self) -> Dict[str, Any]:
        """Return optimizer hyperparameters for serialization."""
        raise NotImplementedError

    def _set_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Restore optimizer hyperparameters from a state dict."""
        raise NotImplementedError

    def _serialize_param_state(self, p: "Tensor") -> Dict[str, Any]:
        """Serialize the per-parameter state for parameter ``p``."""
        raise NotImplementedError

    def _deserialize_param_state(self, p: "Tensor", state: Dict[str, Any]) -> None:
        """Load the per-parameter state for parameter ``p``."""
        raise NotImplementedError
    
class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum, dampening,
    weight decay, and Nesterov momentum.

    Parameters
    ----------
    params : Iterable[Tensor]
        Parameters to optimize.
    lr : float, default=0.001
        Learning rate.
    momentum : float, default=0.0
        Momentum factor.
    dampening : float, default=0.0
        Dampening for momentum.
    weight_decay : float, default=0.0
        L2 penalty (added to the gradient).
    nesterov : bool, default=False
        If True, enables Nesterov momentum (requires ``momentum > 0``).

    Notes
    -----
    - This implementation follows the common PyTorch-style update:
      weight decay is applied by adding ``weight_decay * p.data`` to the gradient.
    - Per-parameter momentum buffers are stored in ``self.state[p]``.
    """
    def __init__(
        self,
        params: Iterable["Tensor"],
        lr: float = 0.001,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.state = {}

    def step(self) -> None:
        """Update parameters in-place using SGD."""
        for p in self.params:
            if p.grad is None:
                continue

            d_p = p.grad.copy()

            if self.weight_decay > 0:
                d_p += self.weight_decay * p.data

            if self.momentum > 0:
                buf = self.state.get(p)
                if buf is None:
                    buf = d_p.copy()
                else:
                    buf *= self.momentum
                    buf += (1 - self.dampening) * d_p
                self.state[p] = buf

                if self.nesterov:
                    d_p += self.momentum * buf
                else:
                    d_p = buf

            p.data -= self.lr * d_p

    def _get_hyperparams(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "dampening": self.dampening,
            "weight_decay": self.weight_decay,
            "nesterov": self.nesterov,
        }
    
    def _set_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        self.lr = hyperparams["lr"]
        self.momentum = hyperparams["momentum"]
        self.dampening = hyperparams["dampening"]
        self.weight_decay = hyperparams["weight_decay"]
        self.nesterov = hyperparams["nesterov"]
    
    def _serialize_param_state(self, p: "Tensor") -> Dict[str, Any]:
        buf = self.state.get(p)
        return {
            "momentum_buffer": buf.copy() if buf is not None else None
        }
    
    def _deserialize_param_state(self, p: "Tensor", state: Dict[str, Any]) -> None:
        if state["momentum_buffer"] is not None:
            self.state[p] = state["momentum_buffer"].copy()

class Adam(Optimizer):
    """
    Adam optimizer with optional weight decay and AMSGrad.

    Parameters
    ----------
    params : Iterable[Tensor]
        Parameters to optimize.
    lr : float, default=0.001
        Learning rate.
    betas : tuple[float, float], default=(0.9, 0.999)
        Coefficients used for computing running averages of gradient and its square.
    eps : float, default=1e-8
        Term added to the denominator for numerical stability.
    weight_decay : float, default=0.0
        L2 penalty added to the gradient (coupled weight decay).
    amsgrad : bool, default=False
        If True, uses the AMSGrad variant.

    Notes
    -----
    - This implements the *coupled* weight decay variant (adds ``weight_decay * p.data``
      to the gradient). If you want decoupled decay, use :class:`AdamW`.
    - State per parameter:
      ``step``, ``exp_avg`` (m), ``exp_avg_sq`` (v), and optionally ``max_exp_avg_sq``.
    """
    def __init__(
        self,
        params: Iterable["Tensor"],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.state = {}

    def step(self) -> None:
        """Update parameters in-place using Adam."""
        for p in self.params:
            if p.grad is None:
                continue

            xp = p.backend
            if p not in self.state:
                self.state[p] = {
                    "step": 0,
                    "exp_avg": xp.zeros_like(p.data),
                    "exp_avg_sq": xp.zeros_like(p.data),
                }
                if self.amsgrad:
                    self.state[p]["max_exp_avg_sq"] = xp.zeros_like(p.data)

            state = self.state[p]
            d_p = p.grad.copy()

            if self.weight_decay != 0:
                d_p += self.weight_decay * p.data

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            beta1, beta2 = self.betas

            state["step"] += 1
            step = state["step"]

            exp_avg[:] = beta1 * exp_avg + (1 - beta1) * d_p            # exponential moving average of the gradient (m_t)
            exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * d_p**2   # exponential moving average of the squared gradient (v_t)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            exp_avg_hat = exp_avg / bias_correction1
            if self.amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
                xp.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = (max_exp_avg_sq / bias_correction2) ** 0.5 + self.eps
            else:
                denom = (exp_avg_sq / bias_correction2) ** 0.5 + self.eps

            p.data -= self.lr * exp_avg_hat / denom

    def _get_hyperparams(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        }
    
    def _set_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        self.lr = hyperparams["lr"]
        self.betas = tuple(hyperparams["betas"])
        self.eps = hyperparams["eps"]
        self.weight_decay = hyperparams["weight_decay"]
        self.amsgrad = hyperparams["amsgrad"]

    def _serialize_param_state(self, p: "Tensor") -> Dict[str, Any]:
        s = self.state[p]
        result = {
            "step": s["step"],
            "exp_avg": s["exp_avg"].copy(),
            "exp_avg_sq": s["exp_avg_sq"].copy(),
        }
        if self.amsgrad and "max_exp_avg_sq" in s:
            result["max_exp_avg_sq"] = s["max_exp_avg_sq"].copy()
        return result
    
    def _deserialize_param_state(self, p: "Tensor", s: Dict[str, Any]) -> None:
        self.state[p] = {
            "step": s["step"],
            "exp_avg": s["exp_avg"].copy(),
            "exp_avg_sq": s["exp_avg_sq"].copy(),
        }
        if self.amsgrad and s.get("max_exp_avg_sq") is not None:
            self.state[p]["max_exp_avg_sq"] = s["max_exp_avg_sq"].copy()

class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).

    Parameters
    ----------
    params : Iterable[Tensor]
        Parameters to optimize.
    lr : float, default=0.001
        Learning rate.
    betas : tuple[float, float], default=(0.9, 0.999)
        Coefficients used for computing running averages of gradient and its square.
    eps : float, default=1e-8
        Term added to the denominator for numerical stability.
    weight_decay : float, default=0.01
        Decoupled weight decay factor.
    amsgrad : bool, default=False
        If True, uses the AMSGrad variant.

    Notes
    -----
    - Weight decay is applied directly to parameters before the Adam update:
      ``p.data *= (1 - lr * weight_decay)``. This is the *decoupled* form.
    - State per parameter is the same as :class:`Adam`.
    """
    def __init__(
        self,
        params: Iterable["Tensor"],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.state = {}

    def step(self) -> None:
        """Update parameters in-place using AdamW."""
        for p in self.params:
            if p.grad is None:
                continue

            xp = p.backend
            if p not in self.state:
                self.state[p] = {
                    "step": 0,
                    "exp_avg": xp.zeros_like(p.data),
                    "exp_avg_sq": xp.zeros_like(p.data),
                }
                if self.amsgrad:
                    self.state[p]["max_exp_avg_sq"] = xp.zeros_like(p.data)

            state = self.state[p]
            d_p = p.grad.copy()

            p.data *= (1 - self.lr * self.weight_decay)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            beta1, beta2 = self.betas

            state["step"] += 1
            step = state["step"]

            exp_avg[:] = beta1 * exp_avg + (1 - beta1) * d_p            # exponential moving average of the gradient (m_t)
            exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * d_p**2   # exponential moving average of the squared gradient (v_t)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            exp_avg_hat = exp_avg / bias_correction1
            if self.amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
                xp.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = (max_exp_avg_sq / bias_correction2) ** 0.5 + self.eps
            else:
                denom = (exp_avg_sq / bias_correction2) ** 0.5 + self.eps

            p.data -= self.lr * exp_avg_hat / denom

    def _get_hyperparams(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
        }
    
    def _set_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        self.lr = hyperparams["lr"]
        self.betas = tuple(hyperparams["betas"])
        self.eps = hyperparams["eps"]
        self.weight_decay = hyperparams["weight_decay"]
        self.amsgrad = hyperparams["amsgrad"]

    def _serialize_param_state(self, p: "Tensor") -> Dict[str, Any]:
        s = self.state[p]
        result = {
            "step": s["step"],
            "exp_avg": s["exp_avg"].copy(),
            "exp_avg_sq": s["exp_avg_sq"].copy(),
        }
        if self.amsgrad and "max_exp_avg_sq" in s:
            result["max_exp_avg_sq"] = s["max_exp_avg_sq"].copy()
        return result
    
    def _deserialize_param_state(self, p: "Tensor", s: Dict[str, Any]) -> None:
        self.state[p] = {
            "step": s["step"],
            "exp_avg": s["exp_avg"].copy(),
            "exp_avg_sq": s["exp_avg_sq"].copy(),
        }
        if self.amsgrad and s.get("max_exp_avg_sq") is not None:
            self.state[p]["max_exp_avg_sq"] = s["max_exp_avg_sq"].copy()