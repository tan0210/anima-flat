import math


def lr_at(step: int, total: int, lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Learning rate with linear warmup and cosine decay.

    Args:
        step: Zero-based optimizer step.
        total: Total number of steps.
        lr: Peak learning rate.
        warmup: Number of warmup steps.
        min_ratio: Final learning rate as a fraction of `lr`.

    Returns:
        The learning rate for `step`.
    """
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min((step - warmup) / max(1, total - warmup), 1.0)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
