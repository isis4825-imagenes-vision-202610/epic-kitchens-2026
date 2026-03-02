"""Distributed samplers for multi-GPU training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator
from torch.utils.data import Dataset, Sampler

from ekmr.utils.distributed import get_rank, get_world_size


class DistributedRetrievalSampler(Sampler[int]):
    """Distributed sampler for retrieval datasets.

    Ensures each GPU gets a disjoint subset of the data
    while maintaining reproducibility.

    Args:
        dataset: The dataset to sample from.
        shuffle: Whether to shuffle indices.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset[object],
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_replicas = get_world_size()
        self.rank = get_rank()
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """Generate indices for the current rank.

        Returns:
            Iterator over dataset indices.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # Pad to total_size
        padding = self.total_size - len(indices)
        if padding > 0:
            indices += indices[:padding]

        # Subsample for this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples for this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffling.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch
