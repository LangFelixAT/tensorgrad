import pickle
from typing import Any, Optional

def save_checkpoint(
    path: str,
    model: Any,
    optimizer: Optional[Any] = None,
    epoch: Optional[int] = None,
    scheduler: Optional[Any] = None,
) -> None:
    """
    Save a training checkpoint to disk.

    The checkpoint contains the model parameters and, optionally, optimizer
    state, scheduler state, and the current epoch.

    Parameters
    ----------
    path : str
        File path where the checkpoint will be written.
    model : Module
        Model whose parameters will be saved. Must implement ``state_dict()``.
    optimizer : Optimizer or None, default=None
        Optional optimizer whose state will be saved.
    epoch : int or None, default=None
        Optional training epoch to store in the checkpoint.
    scheduler : LRScheduler or None, default=None
        Optional learning rate scheduler whose state will be saved.

    Notes
    -----
    - The checkpoint is serialized using :mod:`pickle`.
    - Only ``state_dict`` representations are stored; objects themselves
      are not pickled.
    """
    checkpoint = {
        "model": model.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(
    path: str,
    model: Any,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
) -> Optional[int]:
    """
    Load a training checkpoint from disk.

    Restores model parameters and, optionally, optimizer and scheduler states.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    model : Module
        Model instance into which parameters will be loaded. Must implement
        ``load_state_dict()``.
    optimizer : Optimizer or None, default=None
        Optional optimizer to restore state into.
    scheduler : LRScheduler or None, default=None
        Optional learning rate scheduler to restore state into.

    Returns
    -------
    int or None
        Stored epoch number if present in the checkpoint; otherwise ``None``.

    Notes
    -----
    - If the checkpoint does not contain optimizer or scheduler state, those
      objects are left unchanged.
    - The caller is responsible for ensuring architectural compatibility
      between the checkpoint and the provided model.
    """
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("epoch", None)