from math import floor
from pathlib import Path
from typing import Callable, Literal, Optional, Union
from lance.torch.data import LanceDataset as _LanceDataset
from lance.torch.data import (
    Iterable,
    maybe_sample,
    pa,
    logging,
    ShardedBatchIterator,
    _buffer_arrow_batches,
    CachedDataset,
)
import torch

Any = object()


def _tensorize(obj: Any, dtype: torch.dtype = torch.float32) -> Any:  # noqa: ANN401
    """Turn lists into tensors.

    Parameters
    ----------
    obj : any object
        If a list, attempt to make a tensor. If not or if it fails,
        return the obj unchanged.


    dtype : torch.dtype
        The datatype of the created tensors.

    Returns
    -------
    Any
        Whatever type the object is, unless its been transformed to
        a PyTorch tensor.
    """
    if not isinstance(obj, list):
        return obj

    try:
        return torch.tensor(obj, dtype=dtype)
    except ValueError:
        pass

    return obj


def _to_batch_dict(
    batch: pa.RecordBatch,
    collate_fn: Callable,
) -> Union[dict[str, torch.Tensor], torch.Tensor]:
    """Convert a pyarrow RecordBatch to torch Tensor."""
    batch
    batch_list = batch.to_pylist()
    out_list = []
    for row in batch_list:
        _dict = {}
        for key, val in row.items():
            _dict[key] = _tensorize(val)
        out_list.append(_dict)
    return collate_fn(out_list)


class LanceDataset(_LanceDataset):
    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, str, Path],
        batch_size: int,
        collate_fn: Callable,
        *args,
        columns: Optional[list[str]] = None,
        filter: Optional[str] = None,
        samples: Optional[int] = 0,
        cache: Optional[Union[str, bool]] = None,
        with_row_id: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shard_granularity: Optional[Literal["fragment", "batch"]] = "fragment",
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            *args,
            columns=columns,
            filter=filter,
            samples=samples,
            cache=cache,
            with_row_id=with_row_id,
            rank=rank,
            world_size=world_size,
            shard_granularity=shard_granularity,
            **kwargs,
        )
        self.collate_fn = collate_fn

    def __len__(self):
        raw_stream = ShardedBatchIterator(
            self.dataset,
            self.rank,
            self.world_size,
            columns=self.columns,
            batch_size=self.batch_size,
            with_row_id=self.with_row_id,
            granularity=self.shard_granularity,
        )
        num_rows = floor(raw_stream._ds.count_rows() / self.batch_size)
        return num_rows

    def __iter__(self):
        stream: Iterable[pa.RecordBatch]
        if self.cached_ds:
            stream = self.cached_ds
        else:
            if self.samples:
                raw_stream = maybe_sample(
                    self.dataset,
                    n=self.samples,
                    columns=self.columns,
                    batch_size=self.batch_size,
                )
            elif self.rank is not None and self.world_size is not None:
                logging.info(
                    "Sharded Torch Dataset: rank=%s, world_size=%s, granularity=%s",
                    self.rank,
                    self.world_size,
                    self.shard_granularity,
                )
                raw_stream = ShardedBatchIterator(
                    self.dataset,
                    self.rank,
                    self.world_size,
                    columns=self.columns,
                    batch_size=self.batch_size,
                    with_row_id=self.with_row_id,
                    granularity=self.shard_granularity,
                )
            else:
                raw_stream = self.dataset.to_batches(
                    columns=self.columns,
                    batch_size=self.batch_size,
                    filter=self.filter,
                    with_row_id=self.with_row_id,
                )

            stream = _buffer_arrow_batches(raw_stream, buffer_size=self.batch_size)

            if self.cache:
                self.cached_ds = CachedDataset(stream, cache=self.cache)
                stream = self.cached_ds

        for batch in stream:
            yield _to_batch_dict(batch, self.collate_fn)
            del batch
