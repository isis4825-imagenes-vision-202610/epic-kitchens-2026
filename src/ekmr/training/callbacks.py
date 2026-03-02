"""Training callbacks for monitoring and early stopping."""

from __future__ import annotations

from loguru import logger


class EarlyStopping:
    """Early stopping callback based on a monitored metric.

    Stops training when the monitored metric has not improved
    for a specified number of epochs.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: One of 'min' or 'max'.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs"
                    " without improvement"
                )

        return self.should_stop


class LRMonitor:
    """Learning rate monitoring callback.

    Logs the current learning rate at each step.
    """

    def __call__(self, optimizer: object) -> dict[str, float]:
        """Get current learning rates from optimizer.

        Args:
            optimizer: The optimizer to monitor.

        Returns:
            Dictionary with learning rates per parameter group.
        """
        from torch.optim import Optimizer

        if not isinstance(optimizer, Optimizer):
            return {}
        rates: dict[str, float] = {}
        for i, group in enumerate(optimizer.param_groups):
            rates[f"lr_group_{i}"] = float(group["lr"])
        return rates


class GradNormLogger:
    """Gradient norm logging callback.

    Computes and logs the total gradient norm of model parameters.
    """

    def __call__(self, model: object) -> float:
        """Compute total gradient norm.

        Args:
            model: The model to compute gradients for.

        Returns:
            Total gradient norm as a float.
        """
        import torch.nn as nn

        if not isinstance(model, nn.Module):
            return 0.0

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
