import math
import os
from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from .lora import lora_parameters, save_lora


def sample_sigmas(
    n: int,
    shift: float = 3.0,
    dist: str = "logit_normal",
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Draw training noise levels.

    Args:
        n: Number of samples.
        shift: Schedule shift; the same convention as `sample.sigmas`.
        dist: Distribution of the unshifted time, `"logit_normal"` or `"uniform"`.
        device: Device of the result.

    Returns:
        Float32 tensor of shape `[n]` with values in `[0, 1)`.

    Raises:
        ValueError: If `dist` is unknown.
    """
    if dist == "logit_normal":
        t = torch.sigmoid(torch.randn(n, device=device))
    elif dist == "uniform":
        t = torch.rand(n, device=device)
    else:
        raise ValueError(f"unknown dist {dist!r}")
    return shift * t / (1 + (shift - 1) * t)


def flow_loss(
    model: nn.Module,
    latents: torch.Tensor,
    context: torch.Tensor,
    uncond: torch.Tensor,
    shift: float = 3.0,
    dist: str = "logit_normal",
    caption_dropout: float = 0.1,
) -> torch.Tensor:
    """Flow-matching loss for one batch.

    The model is trained to predict the velocity `noise - latents` at a random noise level.

    Args:
        model: An `AnimaFlat` model.
        latents: Clean latents, shape `[B, H, W, C]`.
        context: Text embeddings, shape `[B, S, text_dim]`, zero-padded like at inference.
        uncond: Empty-prompt embeddings, shape `[1, S, text_dim]`, same `S` as `context`.
        shift: Schedule shift.
        dist: Noise-level distribution (see `sample_sigmas`).
        caption_dropout: Probability of replacing a sample's context with `uncond`.

    Returns:
        Scalar float32 mean-squared error.
    """
    B = latents.shape[0]
    dt = model.patch.weight.dtype  # type: ignore
    x0 = latents.float()
    eps = torch.randn_like(x0)
    sigma = sample_sigmas(B, shift, dist, x0.device)
    s = sigma.view(B, 1, 1, 1)
    xt = (1 - s) * x0 + s * eps
    if caption_dropout > 0:
        drop = torch.rand(B, device=x0.device) < caption_dropout
        context = torch.where(
            drop[:, None, None], uncond.to(context.dtype).expand_as(context), context
        )
    ctx = model.encode_context(context.to(dt))  # type: ignore
    v = model(xt.to(dt), sigma, ctx).float()  # type: ignore
    return F.mse_loss(v, eps - x0)


def lr_at(
    step: int, total: int, lr: float, warmup: int = 0, min_ratio: float = 0.1
) -> float:
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
    p = (step - warmup) / max(1, total - warmup)
    return lr * (
        min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    )


def _cycle(batches: Iterable[dict]) -> Iterator[dict]:
    while True:
        empty = True
        for b in batches:
            empty = False
            yield b
        if empty:
            raise ValueError("no batches to train on")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    directory: str | os.PathLike,
) -> None:
    """Write the adapter and optimizer state to a directory.

    Args:
        model: Model with adapters attached.
        optimizer: The optimizer.
        step: Number of completed optimizer steps.
        directory: Output directory, created if missing.
    """
    os.makedirs(directory, exist_ok=True)
    save_lora(model, os.path.join(directory, "adapter.safetensors"))
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict()},
        os.path.join(directory, "state.pt"),
    )


def finetune(
    model: nn.Module,
    batches: Iterable[dict],
    uncond: torch.Tensor,
    steps: int,
    lr: float = 2e-5,
    weight_decay: float = 0.0,
    warmup: int = 0,
    min_lr_ratio: float = 0.1,
    accum: int = 1,
    max_grad_norm: float = 1.0,
    shift: float = 3.0,
    dist: str = "logit_normal",
    caption_dropout: float = 0.1,
    checkpoint_dir: str | os.PathLike | None = None,
    save_every: int = 500,
    resume: bool = False,
    on_step: Callable[[int, float], None] | None = None,
) -> int:
    """Train the adapters attached to `model`.

    Only the adapter parameters are updated. Attach adapters first with `add_lora` (or
    `load_lora` when resuming). The model is left in training mode.

    Args:
        model: An `AnimaFlat` model with adapters.
        batches: Iterable of dicts with `"latents"` `[B, H, W, C]` and `"context"` `[B, S, text_dim]`
            tensors; restarted when exhausted. All samples within a batch must share a shape.
        uncond: Empty-prompt embeddings, shape `[1, S, text_dim]`.
        steps: Total optimizer steps.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup: Warmup steps.
        min_lr_ratio: Final learning rate as a fraction of `lr`.
        accum: Batches per optimizer step.
        max_grad_norm: Gradient clipping threshold; 0 disables clipping.
        shift: Schedule shift.
        dist: Noise-level distribution (see `sample_sigmas`).
        caption_dropout: Probability of dropping a sample's caption.
        checkpoint_dir: If given, a checkpoint is written here every `save_every` steps and at the end.
        save_every: Steps between checkpoints.
        resume: Restore optimizer state and step count from `checkpoint_dir`; the adapter itself
            must already be loaded.
        on_step: Called as `on_step(step, loss)` after each optimizer step.

    Returns:
        The number of completed steps.

    Raises:
        ValueError: If the model has no adapters, or `batches` is empty.
        FileNotFoundError: If `resume` is set but no checkpoint exists.
    """
    params = lora_parameters(model)
    if not params:
        raise ValueError("model has no adapters")
    device = params[0].device
    uncond = uncond.to(device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    step = 0
    if resume:
        if checkpoint_dir is None:
            raise FileNotFoundError("resume requires checkpoint_dir")
        state = torch.load(
            os.path.join(checkpoint_dir, "state.pt"), map_location=device
        )
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
    model.train()
    it = _cycle(batches)
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, warmup, min_lr_ratio)
        total = 0.0
        for _ in range(accum):
            batch = next(it)
            loss = flow_loss(
                model,
                batch["latents"].to(device),
                batch["context"].to(device),
                uncond,
                shift,
                dist,
                caption_dropout,
            )
            (loss / accum).backward()
            total += loss.item() / accum
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if on_step is not None:
            on_step(step, total)
        if checkpoint_dir is not None and (step % save_every == 0 or step == steps):
            save_checkpoint(model, opt, step, checkpoint_dir)
    return step
