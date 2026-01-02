from typing import Any, Dict, Sequence
import numpy as np

class LRScheduler:
    """
    Base class for learning rate schedulers.

    A scheduler updates the learning rate of an optimizer over time by modifying
    ``optimizer.lr``. Subclasses implement :meth:`step`.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be scheduled. The optimizer is expected
        to expose a mutable attribute ``lr`` (float).

    Notes
    -----
    This is a minimal, PyTorch-inspired interface. State serialization via
    :meth:`state_dict` excludes the optimizer reference.
    """
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
    def step(self, *args: Any, **kwargs: Any) -> None:
        """
        Advance the scheduler by one step.
        """
        raise NotImplementedError
    def state_dict(self) -> Dict[str, Any]:
        """
        Return scheduler state as a Python dictionary.

        Returns
        -------
        dict
            Scheduler attributes excluding the optimizer reference.
        """
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Load scheduler state from a dictionary produced by :meth:`state_dict`.

        Parameters
        ----------
        state : dict
            Scheduler state to restore.
        """
        self.__dict__.update(state)

class StepLR(LRScheduler):
    """
    Decay the learning rate by ``gamma`` every ``step_size`` steps.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule.
    step_size : int
        Period of learning rate decay.
    gamma : float, default=0.1
        Multiplicative decay factor.
    last_epoch : int, default=0
        Initial epoch/step counter (useful when resuming).

    Notes
    -----
    At steps where ``last_epoch % step_size == 0``, the learning rate is updated:
    ``lr *= gamma``.
    """
    def __init__(
        self,
        optimizer: Any,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = 0
    ) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

    def step(self) -> None:
        """Advance by one step and apply decay if at a step boundary."""
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

class MultiStepLR(LRScheduler):
    """
    Decay the learning rate by ``gamma`` at each milestone step.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule.
    milestones : Sequence[int]
        Steps at which to decay the learning rate. Stored sorted.
    gamma : float, default=0.1
        Multiplicative decay factor.
    last_epoch : int, default=0
        Initial epoch/step counter.

    Notes
    -----
    If ``last_epoch`` reaches a value in ``milestones``, the learning rate is updated:
    ``lr *= gamma``.
    """
    def __init__(
        self,
        optimizer: Any,
        milestones: Sequence[int],
        gamma: float = 0.1,
        last_epoch: int = 0,
    ) -> None:
        super().__init__(optimizer)
        self.milestones = sorted(m for m in milestones)
        self.gamma = gamma
        self.last_epoch = last_epoch

    def step(self) -> None:
        """Advance by one step and apply decay if ``last_epoch`` is a milestone."""
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            self.optimizer.lr *= self.gamma

class ExponentialLR(LRScheduler):
    """
    Decay the learning rate exponentially: ``lr = base_lr * gamma**t``.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule.
    gamma : float
        Multiplicative factor per step.
    last_epoch : int, default=0
        Initial epoch/step counter.

    Notes
    -----
    This scheduler stores ``base_lr`` at construction time and recomputes the
    learning rate each step from ``base_lr`` and ``last_epoch``.
    """
    def __init__(
        self,
        optimizer: Any,
        gamma: float,
        last_epoch: int = 0
    ) -> None:
        super().__init__(optimizer)
        self.gamma = gamma
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch

    def step(self) -> None:
        self.last_epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** self.last_epoch)

class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate schedule.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule.
    T_max : int
        Maximum number of steps for one cosine cycle.
    eta_min : float, default=0.0
        Minimum learning rate.
    last_epoch : int, default=0
        Initial epoch/step counter.

    Notes
    -----
    This implements:
    ``lr = eta_min + 0.5*(base_lr - eta_min)*(1 + cos(pi * t / T_max))``
    where ``t`` is the current step clamped to ``[0, T_max]``.
    """
    def __init__(
        self,
        optimizer: Any,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = 0
    ) -> None:
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch

    def step(self) -> None:
        self.last_epoch += 1
        t = min(self.last_epoch, self.T_max)
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * t / self.T_max))

class ReduceLROnPlateau(LRScheduler):
    """
    Reduce learning rate when a monitored metric has stopped improving.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule.
    factor : float, default=0.1
        Multiplicative factor to reduce LR: ``lr *= factor``.
    patience : int, default=5
        Number of bad steps allowed before reducing LR.
    threshold : float, default=1e-4
        Minimum absolute improvement required to reset bad step counter.
    min_lr : float, default=0.0
        Lower bound on the learning rate.
    cooldown : int, default=0
        Number of steps to wait after an LR reduction before resuming normal operation.

    Notes
    -----
    - This is a minimal variant monitoring a scalar metric where *lower is better*
      (e.g. loss). Improvement is defined as ``metric < best - threshold``.
    - Internal counters:
        - ``bad``: consecutive non-improving steps
        - ``cool``: remaining cooldown steps
    """
    def __init__(
        self,
        optimizer: Any,
        factor: float = 0.1,
        patience: int = 5,
        threshold: float = 1e-4,
        min_lr: float = 0.0,
        cooldown: int = 0,
    ) -> None:
        super().__init__(optimizer)
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.best = None
        self.bad = 0
        self.cool = 0

    def step(self, metric: float) -> None:
        """
        Update scheduler based on the provided metric.

        Parameters
        ----------
        metric : float
            Monitored value (lower is considered better).
        """
        if self.cool > 0:
            self.cool -= 1
        improved = (self.best is None) or (metric < self.best - self.threshold)
        if improved:
            self.best = metric
            self.bad = 0
        else:
            self.bad += 1
            if self.bad > self.patience and self.cool == 0:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if new_lr < self.optimizer.lr:
                    self.optimizer.lr = new_lr
                    self.cool = self.cooldown
                self.bad = 0

class WarmupCosineLR(LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer to schedule.
    warmup_steps : int
        Number of warmup steps.
    T_max : int
        Total number of steps (including warmup) for the cosine schedule.
    eta_min : float, default=0.0
        Minimum learning rate for cosine phase.
    lr_warmup_start : float, default=1e-8
        Starting learning rate for warmup.

    Notes
    -----
    - Warmup phase (t <= warmup_steps): linear ramp from ``lr_warmup_start`` to ``base_lr``.
    - Cosine phase: cosine decay from ``base_lr`` to ``eta_min`` over the remaining steps.
    """
    def __init__(
        self,
        optimizer: Any,
        warmup_steps: int,
        T_max: int,
        eta_min: float = 0.0,
        lr_warmup_start: float = 1e-8,
    ) -> None:
        super().__init__(optimizer)
        self.base_lr = optimizer.lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        self.lr_warmup_start = lr_warmup_start
        self.t = 0

    def step(self) -> None:
        """Advance by one step and update learning rate using warmup + cosine decay."""
        self.t += 1
        if self.t <= self.warmup_steps:
            a = (self.base_lr - self.lr_warmup_start) / max(1, self.warmup_steps)
            self.optimizer.lr = self.lr_warmup_start + a * self.t
        else:
            tw = self.t - self.warmup_steps
            T = max(1, self.T_max - self.warmup_steps)
            self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * min(tw, T) / T))