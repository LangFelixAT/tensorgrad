from typing import Any, Optional
from tensorgrad.tensor import no_grad
from tensorgrad.lr_scheduler import ReduceLROnPlateau

def train_one_epoch(
    model: Any,
    dataloader: Any,
    loss_fn: Any,
    optimizer: Any,
) -> float:
    """
    Train a model for one epoch.

    Iterates over the dataloader, performs forward/backward passes, and updates
    parameters using the provided optimizer.

    Parameters
    ----------
    model : Module
        Model to train. Must implement ``train()``, ``__call__``/``forward``, and
        produce a tensor output.
    dataloader : DataLoader
        Iterable yielding batches. Expected to yield ``(x, y)``.
    loss_fn : Module or callable
        Loss function mapping ``(pred, target) -> Tensor``.
    optimizer : Optimizer
        Optimizer with ``zero_grad()``, ``step()``, and attribute ``lr``.

    Returns
    -------
    float
        Mean loss over the epoch (weighted by batch size).
    """
    model.train()
    total_loss = 0.
    total_samples = 0

    for x, y in dataloader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        # Assumes loss is scalar and x is batched along dim 0.
        bs = x.shape[0]
        total_loss += float(loss.data.item()) * bs
        total_samples += bs

    return total_loss / max(1, total_samples)

def evaluate(
    model: Any,
    dataloader: Any,
    loss_fn: Any,
) -> float:
    """
    Evaluate a model without gradient tracking.

    Parameters
    ----------
    model : Module
        Model to evaluate. Must implement ``eval()`` and be callable.
    dataloader : DataLoader
        Iterable yielding batches ``(x, y)``.
    loss_fn : Module or callable
        Loss function mapping ``(pred, target) -> Tensor``.

    Returns
    -------
    float
        Mean loss over the evaluation set (weighted by batch size).
    """
    model.eval()
    total_loss = 0.
    total_samples = 0

    with no_grad():
        for x, y in dataloader:
            out = model(x)
            loss = loss_fn(out, y)
            
            bs = x.shape[0]
            total_loss += float(loss.data.item()) * bs
            total_samples += bs

    return total_loss / total_samples

def accuracy(model: Any, dataloader: Any) -> float:
    model.eval()
    correct = 0
    total = 0

    with no_grad():
        for x, y in dataloader:
            logits = model(x)
            pred = logits.data.argmax(axis=1)
            y_true = y.data.astype(int).reshape(-1)

            correct += int((pred == y_true).sum())
            total += int(y_true.shape[0])

    return 100.0 * correct / max(1, total)

def fit(
    model: Any,
    train_loader: Any,
    loss_fn: Any,
    optimizer: Any,
    num_epochs: int = 10,
    val_loader: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    compute_accuracy: bool = True,
) -> dict:
    """
    Train a model for multiple epochs with optional validation, accuracy tracking,
    and learning-rate scheduling.

    Parameters
    ----------
    model : Module
        Model to train.
    train_loader : DataLoader
        Training data loader yielding batches ``(x, y)``.
    loss_fn : Module or callable
        Loss function mapping ``(pred, target) -> Tensor``.
    optimizer : Optimizer
        Optimizer used for parameter updates.
    num_epochs : int, default=10
        Number of epochs to train.
    val_loader : DataLoader or None, default=None
        Optional validation loader. If provided, validation loss and accuracy
        are computed each epoch.
    scheduler : LRScheduler or None, default=None
        Optional learning-rate scheduler. If the scheduler is an instance of
        :class:`ReduceLROnPlateau`, it is stepped with the monitored metric
        (validation loss if available, else training loss). Otherwise it is
        stepped once per epoch with no arguments.
    compute_accuracy : bool, default=True
        If True, compute classification accuracy on the training loader and
        validation loader (if provided) each epoch.

    Returns
    -------
    dict
        Training history containing per-epoch metrics with keys:
        ``"train_loss"``, ``"val_loss"``, ``"train_acc"``, and ``"val_acc"``.
        Values are lists of floats (or ``None`` where not applicable).

    Notes
    -----
    - This function prints a progress line per epoch including current learning rate,
      loss, and (optionally) accuracy.
    - Assumes the optimizer exposes a float attribute ``lr``.
    - Accuracy computation assumes a classification task where predictions are
      obtained via ``argmax`` over the model outputs.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, loss_fn)
        else:
            val_loss = None

        if compute_accuracy:
            train_acc = accuracy(model, train_loader)
            val_acc = accuracy(model, val_loader) if val_loader is not None else None
        else:
            train_acc = val_acc = None

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_str = f" lr={optimizer.lr:.6g}"
        msg = f"Epoch {epoch+1}/{num_epochs},{lr_str} Train: {train_loss:.4f}"
        if train_acc is not None:
            msg += f" ({train_acc:.2f}%)"
        if val_loss is not None:
            msg += f"  Val: {val_loss:.4f}"
        if val_acc is not None:
            msg += f" ({val_acc:.2f}%)"
        print(msg)

    return history