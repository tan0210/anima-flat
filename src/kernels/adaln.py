import torch
import triton
from triton import language as tl

from .common import _check

_ROW_CONFIGS = [triton.Config({}, num_warps=nw) for nw in (4, 8, 16)]


@triton.autotune(configs=_ROW_CONFIGS, key=["D"])
@triton.jit
def _residual_adaln_kernel(
    x_ptr,
    res_ptr,
    gate_ptr,
    shift_ptr,
    scale_ptr,
    h_ptr,
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
    tl.store(h_ptr + off, (d * rstd * sc + sh).to(h_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op("anima_flat::residual_adaln", mutates_args=())
def _residual_adaln(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, D = residual.shape
    h = torch.empty_like(residual)
    z = torch.empty_like(residual)
    _residual_adaln_kernel[(B * S,)](
        x,
        residual,
        gate,
        shift,
        scale,
        h,
        z,
        S,
        eps,
        D=D,
        BLOCK=triton.next_power_of_2(D),
        HAS_X=True,
    )
    return h, z


@_residual_adaln.register_fake
def _(x, residual, gate, shift, scale, eps):
    return torch.empty_like(residual), torch.empty_like(residual)


@torch.library.custom_op("anima_flat::adaln", mutates_args=())
def _adaln(
    z: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float
) -> torch.Tensor:
    B, S, D = z.shape
    h = torch.empty_like(z)
    _residual_adaln_kernel[(B * S,)](
        z,
        z,
        shift,
        shift,
        scale,
        h,
        h,
        S,
        eps,
        D=D,
        BLOCK=triton.next_power_of_2(D),
        HAS_X=False,
    )
    return h


@_adaln.register_fake
def _(z, shift, scale, eps):
    return torch.empty_like(z)


def residual_adaln(x, residual, gate, shift, scale, eps=1e-6):
    _check(x, residual, gate, shift, scale)
    if x.shape != residual.shape or x.dtype != residual.dtype:
        raise ValueError("x and residual must match")
    return _residual_adaln(
        x.contiguous(),
        residual.contiguous(),
        gate.contiguous(),
        shift.contiguous(),
        scale.contiguous(),
        eps,
    )


def adaln(z, shift, scale, eps=1e-6):
    _check(z, shift, scale)
    return _adaln(z.contiguous(), shift.contiguous(), scale.contiguous(), eps)
