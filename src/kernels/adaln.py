import torch
import triton
from triton import language as tl

from .common import _check, tuned_rows

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


@triton.autotune(configs=_ROW_CONFIGS, key=["D", "ROWS"])
@triton.jit
def _residual_adaln_bwd_kernel(
    gh_ptr,
    gz_ptr,
    z_ptr,
    x_ptr,
    gate_ptr,
    scale_ptr,
    dz_ptr,
    dx_ptr,
    psh_ptr,
    psc_ptr,
    pg_ptr,
    S,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_X: tl.constexpr,
    HAS_GZ: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    nchunks = tl.num_programs(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    sc = tl.load(scale_ptr + batch * D + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_X:
        g = tl.load(gate_ptr + batch * D + cols, mask=mask, other=0.0).to(tl.float32)
        acc_g = tl.zeros([BLOCK], dtype=tl.float32)
    acc_sh = tl.zeros([BLOCK], dtype=tl.float32)
    acc_sc = tl.zeros([BLOCK], dtype=tl.float32)
    for r in range(ROWS):
        s = chunk * ROWS + r
        m = cols < tl.where(s < S, D, 0)
        off = (batch * S + s) * D + cols
        z = tl.load(z_ptr + off, mask=m, other=0.0).to(tl.float32)
        gh = tl.load(gh_ptr + off, mask=m, other=0.0).to(tl.float32)
        mean = tl.sum(z, axis=0) / D
        d = tl.where(m, z - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(d * d, axis=0) / D + eps)
        xhat = d * rstd
        acc_sh += gh
        acc_sc += gh * xhat
        dxh = gh * sc
        c1 = tl.sum(dxh, axis=0) / D
        c2 = tl.sum(dxh * xhat, axis=0) / D
        dz = rstd * (dxh - c1 - xhat * c2)
        if HAS_GZ:
            dz += tl.load(gz_ptr + off, mask=m, other=0.0).to(tl.float32)
        dz = tl.where(m, dz, 0.0)
        tl.store(dz_ptr + off, dz.to(dz_ptr.dtype.element_ty), mask=m)
        if HAS_X:
            x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
            tl.store(dx_ptr + off, (dz * g).to(dx_ptr.dtype.element_ty), mask=m)  # type: ignore
            acc_g += dz * x  # type: ignore
    part = (batch * nchunks + chunk) * D + cols
    tl.store(psh_ptr + part, acc_sh, mask=mask)
    tl.store(psc_ptr + part, acc_sc, mask=mask)
    if HAS_X:
        tl.store(pg_ptr + part, acc_g, mask=mask)  # type: ignore


def _residual_adaln_bwd(gh, gz, z, x, gate, scale, eps):
    B, S, D = z.shape
    has_x = x is not None
    gh = gh.contiguous()
    gz = gz.contiguous() if gz is not None else None

    def launch(rows):
        nchunks = triton.cdiv(S, rows)
        dz = torch.empty_like(z)
        dx = torch.empty_like(z) if has_x else dz
        partial = lambda: torch.empty(
            B, nchunks, D, dtype=torch.float32, device=z.device
        )
        psh, psc = partial(), partial()
        pg = partial() if has_x else psh
        _residual_adaln_bwd_kernel[(B, nchunks)](
            gh,
            gz if gz is not None else gh,
            z,
            x if has_x else z,
            gate if has_x else scale,
            scale,
            dz,
            dx,
            psh,
            psc,
            pg,
            S,
            eps,
            D=D,
            BLOCK=triton.next_power_of_2(D),
            ROWS=rows,
            HAS_X=has_x,
            HAS_GZ=gz is not None,
        )
        return dz, dx, psh, psc, pg

    rows = tuned_rows(("adaln_bwd", B, S, D, has_x, gz is not None, z.dtype), launch)
    dz, dx, psh, psc, pg = launch(rows)
    dshift = psh.sum(1).to(scale.dtype).view_as(scale)
    dscale = psc.sum(1).to(scale.dtype).view_as(scale)
    dgate = pg.sum(1).to(gate.dtype).view_as(gate) if has_x else None
    return dz, dx if has_x else None, dgate, dshift, dscale


def _residual_adaln_setup(ctx, inputs, output):
    x, _, gate, _, scale, eps = inputs
    ctx.save_for_backward(x, gate, scale, output[1])
    ctx.eps = eps


def _residual_adaln_backward(ctx, gh, gz):
    x, gate, scale, z = ctx.saved_tensors
    dres, dx, dgate, dshift, dscale = _residual_adaln_bwd(
        gh, gz, z, x, gate, scale, ctx.eps
    )
    return dx, dres, dgate, dshift, dscale, None


torch.library.register_autograd(
    "anima_flat::residual_adaln",
    _residual_adaln_backward,
    setup_context=_residual_adaln_setup,
)


def _adaln_setup(ctx, inputs, output):
    z, _, scale, eps = inputs
    ctx.save_for_backward(z, scale)
    ctx.eps = eps


def _adaln_backward(ctx, gh):
    z, scale = ctx.saved_tensors
    dz, _, _, dshift, dscale = _residual_adaln_bwd(
        gh, None, z, None, None, scale, ctx.eps
    )
    return dz, dshift, dscale, None


torch.library.register_autograd(
    "anima_flat::adaln", _adaln_backward, setup_context=_adaln_setup
)


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
