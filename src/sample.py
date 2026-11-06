import math
from collections.abc import Callable, Sequence

import numpy as np
import torch


def sigmas(n: int, shift: float = 3.0) -> torch.Tensor:
    """Build the shifted flow-matching noise schedule.

    Args:
        n: Number of denoising steps.
        shift: Schedule shift; larger values spend more steps at high noise.

    Returns:
        Float32 tensor of shape `[n + 1]`, decreasing from near 1 to a final 0.
    """
    s = torch.linspace(1.0, 1.0 / n, n, dtype=torch.float64)
    s = shift * s / (1 + (shift - 1) * s)
    return torch.cat([s, s.new_zeros(1)]).float()


def noise(
    batch: int,
    h: int,
    w: int,
    channels: int,
    seed: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Draw reproducible Gaussian latent noise.

    Args:
        batch: Batch size.
        h: Latent height.
        w: Latent width.
        channels: Latent channels.
        seed: RNG seed.
        device: Device for the generator and result.

    Returns:
        Float32 tensor of shape `[batch, h, w, channels]`.
    """
    g = torch.Generator(device).manual_seed(seed)
    return torch.randn(batch, h, w, channels, generator=g, device=device)


@torch.no_grad()
def decode(
    vae: torch.nn.Module,
    mean: Sequence[float] | torch.Tensor,
    std: Sequence[float] | torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Decode a latent to an image with the VAE.

    Args:
        vae: Video-style VAE exposing `.dtype` and `.decode(z)`.
        mean: Per-channel latent mean, length `C`.
        std: Per-channel latent std, length `C`.
        x: Latent, shape `[B, H, W, C]`.

    Returns:
        Decoded images, shape `[B, 3, H', W']` (first and only frame of the VAE output).
    """
    z = x.permute(0, 3, 1, 2).unsqueeze(2)
    mean = torch.as_tensor(mean, device=z.device, dtype=torch.float32).view(
        1, -1, 1, 1, 1
    )
    std = torch.as_tensor(std, device=z.device, dtype=torch.float32).view(
        1, -1, 1, 1, 1
    )
    z = z * std + mean
    z = z.to(vae.dtype)  # type: ignore
    return vae.decode(z)[0][:, :, 0]  # type: ignore


SAMPLERS = ("euler", "er_sde")
NOISE_SCALES = ("er", "ode")


def euler(
    velocity: Callable[[int, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    sg: torch.Tensor,
) -> torch.Tensor:
    """Integrate the flow with Euler steps.

    Args:
        velocity: Maps `(step index, latent)` to the predicted velocity, same shape as the latent.
        x: Initial noise latent.
        sg: Noise schedule from `sigmas`, on the same device as `x`.

    Returns:
        The denoised latent.
    """
    for i in range(len(sg) - 1):
        x = x + (sg[i + 1] - sg[i]) * velocity(i, x)
    return x


def _er_phi(lam: np.ndarray) -> np.ndarray:
    return lam * (np.exp(lam**0.3) + 10.0)


def _ode_phi(lam: np.ndarray) -> np.ndarray:
    return lam


def _er_integrals(
    lam_t: float, lam_s: float, phi: Callable[[np.ndarray], np.ndarray]
) -> tuple[float, float]:
    # I1 = int_{lam_t}^{lam_s} 1/phi, I2 = int_{lam_t}^{lam_s} (lam - lam_s)/phi, by Gauss-Legendre
    # in log(lam) where the integrand is smooth.
    nodes, w = np.polynomial.legendre.leggauss(64)
    a, b = math.log(lam_t), math.log(lam_s)
    u = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    lam = np.exp(u)
    scale = 0.5 * (b - a) * w * lam / phi(lam)
    return float(scale.sum()), float((scale * (lam - lam_s)).sum())


def er_sde_plan(
    sg: Sequence[float] | torch.Tensor,
    max_stage: int = 3,
    s_noise: float = 1.0,
    phi: Callable[[np.ndarray], np.ndarray] = _er_phi,
    first_offset: float = 1e-4,
) -> list[tuple[float, tuple[float, float, float], float]]:
    """Precompute the per-step coefficients of the ER-SDE solver.

    The coefficients depend only on the schedule, so they are computed once in float64.

    Args:
        sg: Noise schedule from `sigmas`, length `steps + 1`, ending in 0.
        max_stage: Highest solver order used (1 to 3); step `i` uses `min(max(i, 1), max_stage)`.
        s_noise: Multiplier on the injected noise.
        phi: Noise-scale function of `lambda = sigma / (1 - sigma)`; `_ode_phi` gives the
            noise-free ODE solver.
        first_offset: The first sigma is clamped to `1 - first_offset`, since `lambda` is infinite
            at sigma 1.

    Returns:
        One `(c_x, (w_0, w_1, w_2), noise_std)` per step: the new latent is
        `c_x * x + w_0 * x0 + w_1 * x0_prev + w_2 * x0_prev2 + noise_std * z` where `x0` is the
        current clean-latent prediction and `z` is standard normal.

    Raises:
        ValueError: If `max_stage` is not 1, 2 or 3.
    """
    if max_stage not in (1, 2, 3):
        raise ValueError("max_stage must be 1, 2 or 3")
    sig = [
        min(float(v), 1.0 - first_offset) for v in torch.as_tensor(sg).double().tolist()
    ]
    n = len(sig) - 1
    alpha = [1.0 - v for v in sig]
    lam = [v / a for v, a in zip(sig, alpha, strict=True)]
    plan = []
    for i in range(n):
        if sig[i + 1] <= 0.0:
            plan.append((0.0, (1.0, 0.0, 0.0), 0.0))
            continue
        ls, lt, at, as_ = lam[i], lam[i + 1], alpha[i + 1], alpha[i]
        pt, ps = float(phi(np.array(lt))), float(phi(np.array(ls)))
        r = pt / ps
        c_x = at / as_ * r
        std = (
            0.0
            if phi is _ode_phi
            else s_noise * at * math.sqrt(max(lt * lt - r * r * ls * ls, 0.0))
        )
        w = [at * (1.0 - r), 0.0, 0.0]
        # The first prediction is made at pure noise (lambda is infinite) and is not used as history.
        stage = min(max(i, 1), max_stage)
        if stage >= 2:
            i1, i2 = _er_integrals(lt, ls, phi)
            a1 = at * ((lt - ls) + pt * i1)
            d1 = ls - lam[i - 1]
            if stage == 2:
                w[0] += a1 / d1
                w[1] -= a1 / d1
            else:
                a2 = at * ((lt - ls) ** 2 / 2 + pt * i2)
                d1b = lam[i - 1] - lam[i - 2]
                big = d1 + d1b
                # Newton form through the last three predictions:
                # x0' = f1 + d1 * f2 and x0'' = 2 * f2 at the current point.
                f1 = (1 / d1, -1 / d1, 0.0)
                f2 = (1 / (d1 * big), -(1 / d1 + 1 / d1b) / big, 1 / (d1b * big))
                for k in range(3):
                    w[k] += a1 * (f1[k] + d1 * f2[k]) + a2 * 2 * f2[k]
        plan.append((c_x, (w[0], w[1], w[2]), std))
    return plan


def er_sde(
    velocity: Callable[[int, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    sg: torch.Tensor,
    max_stage: int = 3,
    s_noise: float = 1.0,
    seeds: Sequence[int] | None = None,
    noise_scale: str = "er",
) -> torch.Tensor:
    """Sample with the ER-SDE solver (Cui et al., arXiv:2309.06169) for flow models.

    Uses one model evaluation per step.

    Args:
        velocity: Maps `(step index, latent)` to the predicted velocity, same shape as the latent.
        x: Initial noise latent, float32, shape `[B, ...]`.
        sg: Noise schedule from `sigmas`.
        max_stage: Highest solver order (1 to 3).
        s_noise: Multiplier on the injected noise. Values below 1 shrink the result (0 collapses
            it to the average prediction); for a noise-free sampler use `noise_scale="ode"`.
        seeds: One seed per batch element for the injected noise, so a sample does not depend on
            the rest of the batch; if None, the global torch RNG is used.
        noise_scale: `"er"` for the stochastic solver, or `"ode"` for its deterministic
            (noise-free) limit, a multistep ODE solver of the same order.

    Returns:
        The denoised latent.

    Raises:
        ValueError: If `max_stage` is not 1, 2 or 3, `noise_scale` is unknown, or `seeds` does not
            match the batch size.
    """
    if noise_scale not in NOISE_SCALES:
        raise ValueError(
            f"unknown noise_scale {noise_scale!r}; expected one of {NOISE_SCALES}"
        )
    plan = er_sde_plan(
        sg, max_stage, s_noise, _ode_phi if noise_scale == "ode" else _er_phi
    )
    sig = sg.double().cpu().tolist()
    if seeds is not None and len(seeds) != x.shape[0]:
        raise ValueError("need one seed per batch element")
    gens = None
    if seeds is not None:
        # Offset so the noise stream differs from the one that drew the initial latent.
        gens = [
            torch.Generator(x.device).manual_seed(int(s) + (1 << 31)) for s in seeds
        ]
    hist: list[torch.Tensor] = []
    for i, (c_x, w, std) in enumerate(plan):
        x0 = x - sig[i] * velocity(i, x)
        out = c_x * x + w[0] * x0
        if w[1]:
            out = out + w[1] * hist[-1]
        if w[2]:
            out = out + w[2] * hist[-2]
        if std:
            if gens is None:
                z = torch.randn_like(x)
            else:
                z = torch.cat(
                    [
                        torch.randn(
                            1, *x.shape[1:], generator=g, device=x.device, dtype=x.dtype
                        )
                        for g in gens
                    ]
                )
            out = out + std * z
        hist = (hist + [x0])[-2:]
        x = out
    return x
