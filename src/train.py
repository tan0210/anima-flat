import torch
import torch.nn.functional as F
from torch import nn


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
