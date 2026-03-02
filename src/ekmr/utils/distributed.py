"""Distributed training helpers for DDP."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    """Check if distributed training is initialized."""
    return bool(dist.is_available() and dist.is_initialized())


def get_rank() -> int:
    """Get the rank of the current process."""
    if is_dist_initialized():
        return int(dist.get_rank())
    return 0


def get_world_size() -> int:
    """Get the total number of processes in the distributed group."""
    if is_dist_initialized():
        return int(dist.get_world_size())
    return 1


def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed() -> None:
    """Initialize distributed training from environment variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )


def cleanup_distributed() -> None:
    """Clean up the distributed process group."""
    if is_dist_initialized():
        dist.destroy_process_group()
