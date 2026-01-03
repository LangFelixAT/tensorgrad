from typing import Any, Tuple, Union
from tensorgrad.tensor import Tensor

class Dataset:
    """
    Base class for datasets.

    A dataset provides random access to samples via :meth:`__getitem__`
    and reports its size via :meth:`__len__`. This mirrors the minimal
    interface of ``torch.utils.data.Dataset``.
    """

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        """
        Return a single sample at index ``idx``.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Any
            A single sample. Commonly a :class:`Tensor` or a tuple of tensors
            (e.g. ``(x, y)``).
        """
        raise NotImplementedError
    
class TensorDataset(Dataset):
    """
    Dataset wrapping one or more tensors.

    Each sample is obtained by indexing all tensors along dimension 0.

    Parameters
    ----------
    *tensors : Tensor
        One or more tensors with the same size in dimension 0.

    Notes
    -----
    The dataset length is ``tensors[0].shape[0]``.
    """
    def __init__(self, *tensors: Tensor) -> None:
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """Return the sample tuple at index ``idx``."""
        return tuple(t[idx] for t in self.tensors)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.tensors[0].shape[0]
    
class DataLoader:
    """
    Simple data loader providing batching and optional shuffling.

    Iterates over a :class:`Dataset` and yields mini-batches. If dataset items
    are tuples (e.g. ``(x, y)``), the loader returns a tuple of stacked tensors.
    Otherwise it returns a single stacked tensor.

    Parameters
    ----------
    dataset : Dataset
        Dataset to iterate over.
    batch_size : int, default=1
        Number of samples per batch.
    shuffle : bool, default=False
        If True, shuffles indices at the start of each iteration.
    drop_last : bool, default=False
        If True, drops the last incomplete batch.

    Notes
    -----
    - Stacking is performed using the backend (NumPy/CuPy) inferred from the
      first sample.
    - Returned batches are new tensors created from stacked ``.data`` arrays.
      Gradients are not tracked through the data loading process.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> "DataLoader":
        """
        Initialize the iterator.

        Infers the backend from the first sample, builds the index array,
        optionally shuffles, and resets the batch pointer.
        """
        first_item = self.dataset[0]
        first_tensor = first_item[0] if isinstance(first_item, tuple) else first_item
        self.backend = first_tensor.backend        
        self.indices = self.backend.arange(len(self.dataset))

        if self.shuffle:
            self.backend.random.shuffle(self.indices)
        self.idx = 0
        return self

    def __next__(self) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Return the next mini-batch.

        Returns
        -------
        Tensor or tuple[Tensor, ...]
            If dataset items are tuples, returns a tuple of batched tensors
            (stacked along dim 0). Otherwise returns a single batched tensor.

        Raises
        ------
        StopIteration
            When the iterator is exhausted or when ``drop_last=True`` and the
            remaining samples are fewer than ``batch_size``.
        """
        if self.idx >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.idx:self.idx + self.batch_size]

        if self.drop_last and len(batch_indices) < self.batch_size:
            raise StopIteration

        batch = [self.dataset[int(i)] for i in batch_indices]
        self.idx += self.batch_size

        if isinstance(batch[0], tuple):
            return tuple(Tensor(self.backend.stack([item.data for item in items])) for items in zip(*batch))
        else:
            return Tensor(self.backend.stack([item.data for item in batch]))