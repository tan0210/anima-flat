from typing import NamedTuple

import torch
import triton
from triton import language as tl

from .common import _check

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0

_ROW_CONFIGS = [triton.Config({}, num_warps=nw) for nw in (4, 8, 16)]


class FP8Weight(NamedTuple):
    """A weight quantized to FP8 with one scale per output channel.

    Attributes:
        q: FP8 weight, shape `[N, K]`.
        s: Float32 scales, shape `[N]`.
    """

    q: torch.Tensor
    s: torch.Tensor


def supported(device: torch.device | str | None = None) -> bool:
    """Tell whether the GPU can run the FP8 path (Hopper or newer datacenter parts).

    Args:
        device: CUDA device to check; defaults to the current one.

    Returns:
        True if row-scaled FP8 matrix multiplication is available.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device)[0] in (9, 10)


def quantize_weight(w: torch.Tensor) -> FP8Weight:
    """Quantize a `[N, K]` weight to FP8 with a scale per output channel.

    Args:
        w: Weight of any floating dtype.

    Returns:
        The quantized weight.
    """
    wf = w.float()
    s = (wf.abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12)
    q = (wf / s[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return FP8Weight(q.contiguous(), s.contiguous())


@triton.autotune(configs=_ROW_CONFIGS, key=["D"])
@triton.jit
def _residual_adaln_fp8_kernel(
    x_ptr,
    res_ptr,
    gate_ptr,
    shift_ptr,
    scale_ptr,
    q_ptr,
    s_ptr,
    z_ptr,
    S,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_X: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // S
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    off = row * D + cols
    res = tl.load(res_ptr + off, mask=mask, other=0.0)
    if HAS_X:
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + batch * D + cols, mask=mask, other=0.0).to(tl.float32)
        prod = (g * x).to(res_ptr.dtype.element_ty).to(tl.float32)
        z = (res.to(tl.float32) + prod).to(res_ptr.dtype.element_ty)
        tl.store(z_ptr + off, z, mask=mask)
    else:
        z = res
    zf = z.to(tl.float32)
    mean = tl.sum(zf, axis=0) / D
    d = tl.where(mask, zf - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(d * d, axis=0) / D + eps)
    sc = tl.load(scale_ptr + batch * D + cols, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(shift_ptr + batch * D + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * sc + sh
    row_scale = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-12) / 448.0
    q = tl.minimum(tl.maximum(y / row_scale, -448.0), 448.0)
    tl.store(q_ptr + off, q.to(q_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + row, row_scale)


@torch.library.custom_op("anima_flat::residual_adaln_fp8", mutates_args=())
def _residual_adaln_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, D = residual.shape
    q = torch.empty(B, S, D, dtype=FP8, device=residual.device)
    s = torch.empty(B * S, 1, dtype=torch.float32, device=residual.device)
    z = torch.empty_like(residual)
    _residual_adaln_fp8_kernel[(B * S,)](
        x,
        residual,
        gate,
        shift,
        scale,
        q,
        s,
        z,
        S,
        eps,
        D=D,
        BLOCK=triton.next_power_of_2(D),
        HAS_X=True,
    )
    return q, s, z


@_residual_adaln_fp8.register_fake
def _(x, residual, gate, shift, scale, eps):
    B, S, D = residual.shape
    return (
        torch.empty(B, S, D, dtype=FP8, device=residual.device),
        torch.empty(B * S, 1, dtype=torch.float32, device=residual.device),
        torch.empty_like(residual),
    )


@torch.library.custom_op("anima_flat::adaln_fp8", mutates_args=())
def _adaln_fp8(
    z: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, D = z.shape
    q = torch.empty(B, S, D, dtype=FP8, device=z.device)
    s = torch.empty(B * S, 1, dtype=torch.float32, device=z.device)
    _residual_adaln_fp8_kernel[(B * S,)](
        z,
        z,
        shift,
        shift,
        scale,
        q,
        s,
        z,
        S,
        eps,
        D=D,
        BLOCK=triton.next_power_of_2(D),
        HAS_X=False,
    )
    return q, s


@_adaln_fp8.register_fake
def _(z, shift, scale, eps):
    B, S, D = z.shape
    return (
        torch.empty(B, S, D, dtype=FP8, device=z.device),
        torch.empty(B * S, 1, dtype=torch.float32, device=z.device),
    )


def residual_adaln_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`residual_adaln` whose modulated output is returned as FP8 with a scale per token.

    Args:
        x: Sub-layer output, shape `[B, S, D]`.
        residual: Residual stream, same shape and dtype as `x`.
        gate: Gate, shape `[B, D]`.
        shift: Shift, shape `[B, D]`.
        scale: Scale (already including the +1), shape `[B, D]`.
        eps: LayerNorm epsilon.

    Returns:
        `(q, s, z)`: the FP8 output `[B, S, D]`, its float32 token scales `[B * S, 1]`, and the
        updated residual stream `z` `[B, S, D]` in the input dtype.

    Raises:
        ValueError: If a tensor is not on CUDA or `x` and `residual` differ in shape or dtype.
    """
    _check(x, residual, gate, shift, scale)
    if x.shape != residual.shape or x.dtype != residual.dtype:
        raise ValueError("x and residual must match")
    return _residual_adaln_fp8(
        x.contiguous(),
        residual.contiguous(),
        gate.contiguous(),
        shift.contiguous(),
        scale.contiguous(),
        eps,
    )


def adaln_fp8(
    z: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """`adaln` whose output is returned as FP8 with a scale per token.

    Args:
        z: Tokens, shape `[B, S, D]`.
        shift: Shift, shape `[B, D]`.
        scale: Scale (already including the +1), shape `[B, D]`.
        eps: LayerNorm epsilon.

    Returns:
        `(q, s)`: the FP8 output `[B, S, D]` and its float32 token scales `[B * S, 1]`.

    Raises:
        ValueError: If a tensor is not on CUDA.
    """
    _check(z, shift, scale)
    return _adaln_fp8(z.contiguous(), shift.contiguous(), scale.contiguous(), eps)


def fp8_linear(
    q: torch.Tensor, s: torch.Tensor, w: FP8Weight, fast_accum: bool = False
) -> torch.Tensor:
    """Multiply FP8 tokens by an FP8 weight; scales are applied inside the matmul.

    Args:
        q: FP8 tokens, shape `[B, S, K]`.
        s: Token scales from `adaln_fp8` or `residual_adaln_fp8`, shape `[B * S, 1]`.
        w: Weight from `quantize_weight`, shape `[N, K]`.
        fast_accum: Accumulate with reduced precision for extra speed.

    Returns:
        bfloat16 result, shape `[B, S, N]`.
    """
    B, S, K = q.shape
    out = torch._scaled_mm(  # type: ignore
        q.view(B * S, K),
        w.q.t(),
        scale_a=s,
        scale_b=w.s.view(1, -1),
        out_dtype=torch.bfloat16,
        use_fast_accum=fast_accum,
    )
    return out.view(B, S, -1)


@triton.jit
def _gelu(x):
    z = x * 0.7071067811865476
    az = tl.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    poly = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t
    y = 1.0 - poly * tl.exp(-az * az)
    return 0.5 * x * (1.0 + tl.where(z < 0, -y, y))


@triton.autotune(configs=_ROW_CONFIGS, key=["K"])
@triton.jit
def _rowquant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    GELU: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row * K + cols, mask=mask, other=0.0).to(tl.float32)
    if GELU:
        x = _gelu(x)
    row_scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12) / 448.0
    q = tl.minimum(tl.maximum(x / row_scale, -448.0), 448.0)
    tl.store(q_ptr + row * K + cols, q.to(q_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + row, row_scale)


@torch.library.custom_op("anima_flat::rowquant", mutates_args=())
def _rowquant(x: torch.Tensor, gelu: bool) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, K = x.shape
    q = torch.empty(B, S, K, dtype=FP8, device=x.device)
    s = torch.empty(B * S, 1, dtype=torch.float32, device=x.device)
    _rowquant_kernel[(B * S,)](x, q, s, K=K, BLOCK=triton.next_power_of_2(K), GELU=gelu)
    return q, s


@_rowquant.register_fake
def _(x, gelu):
    B, S, K = x.shape
    return (
        torch.empty(B, S, K, dtype=FP8, device=x.device),
        torch.empty(B * S, 1, dtype=torch.float32, device=x.device),
    )


def rowquant(x: torch.Tensor, gelu: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tokens to FP8 with a scale per token, optionally applying GELU first.

    Args:
        x: Tokens, shape `[B, S, K]`.
        gelu: Apply GELU before quantizing (within 1e-6 of the exact erf GELU).

    Returns:
        `(q, s)`: the FP8 tokens `[B, S, K]` and their float32 scales `[B * S, 1]`, as accepted
        by `fp8_linear`.

    Raises:
        ValueError: If `x` is not on CUDA.
    """
    _check(x)
    return _rowquant(x.contiguous(), gelu)
