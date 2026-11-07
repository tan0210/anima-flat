import torch
import triton
from triton import language as tl

from .common import _check, tuned_rows

_CONFIGS = [triton.Config({}, num_warps=nw) for nw in (2, 4, 8)]


@triton.jit
def _offset(row, H, W, C, SHUF: tl.constexpr):
    if SHUF:
        hw = H * W
        b = row // hw
        rem = row - b * hw
        y = rem // W
        x = rem - y * W
        return (
            ((b * (H // 2) + y // 2) * (W // 2) + x // 2) * 4 + (y % 2) * 2 + x % 2
        ) * C
    return row * C


@triton.autotune(configs=_CONFIGS, key=["C"])
@triton.jit
def _fwd_kernel(
    x_ptr,
    r_ptr,
    g_ptr,
    s_ptr,
    y_ptr,
    H,
    W,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_R: tl.constexpr,
    SHUF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < C
    x = tl.load(x_ptr + _offset(row, H, W, C, SHUF) + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    if HAS_R:
        r = tl.load(r_ptr + row * C + cols, mask=mask, other=0.0).to(tl.float32)
        s = (x + r).to(s_ptr.dtype.element_ty).to(tl.float32)
        tl.store(s_ptr + row * C + cols, s.to(s_ptr.dtype.element_ty), mask=mask)
    elif SHUF:
        s = x
        tl.store(s_ptr + row * C + cols, s.to(s_ptr.dtype.element_ty), mask=mask)
    else:
        s = x
    rstd = (C**0.5) / tl.maximum(tl.sqrt(tl.sum(s * s, axis=0)), 1e-12)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    a = s * rstd * g
    tl.store(
        y_ptr + row * C + cols,
        (a * tl.sigmoid(a)).to(y_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.autotune(configs=_CONFIGS, key=["C", "ROWS"])
@triton.jit
def _bwd_kernel(
    s_ptr,
    g_ptr,
    dy_ptr,
    gs_ptr,
    ds_ptr,
    pg_ptr,
    N,
    H,
    W,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_GS: tl.constexpr,
    SHUF: tl.constexpr,
):
    chunk = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < C
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(ROWS):
        row = (chunk * ROWS + i).to(tl.int64)
        m = cols < tl.where(row < N, C, 0)
        s = tl.load(s_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
        rstd = (C**0.5) / tl.maximum(tl.sqrt(tl.sum(s * s, axis=0)), 1e-12)
        n = s * rstd
        a = n * g
        sg = tl.sigmoid(a)
        da = dy * sg * (1.0 + a * (1.0 - sg))
        acc += da * n
        dn = da * g
        ds = rstd * (dn - n * (tl.sum(dn * n, axis=0) / C))
        if HAS_GS:
            ds += tl.load(gs_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(
            ds_ptr + _offset(row, H, W, C, SHUF) + cols,
            ds.to(ds_ptr.dtype.element_ty),
            mask=m,
        )
    tl.store(pg_ptr + chunk * C + cols, acc, mask=mask)


def _launch_fwd(x, r, g, shuf):
    C = g.numel()
    if shuf:
        B, h, w, _ = x.shape
        shape = (B, 2 * h, 2 * w, C)
        H, W = 2 * h, 2 * w
    else:
        shape, H, W = x.shape, 0, 0
    N = x.numel() // x.shape[-1] if not shuf else shape[0] * shape[1] * shape[2]
    y = torch.empty(shape, dtype=x.dtype, device=x.device)
    s = (
        torch.empty(shape, dtype=x.dtype, device=x.device)
        if (r is not None or shuf)
        else x
    )
    _fwd_kernel[(N,)](
        x,
        r if r is not None else x,
        g,
        s,
        y,
        H,
        W,
        C=C,
        BLOCK=triton.next_power_of_2(C),
        HAS_R=r is not None,
        SHUF=shuf,
    )
    return s, y


@torch.library.custom_op("anima_flat::rms_silu", mutates_args=())
def _rms_silu(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    _check(x)
    return _launch_fwd(x.contiguous(), None, gamma, False)[1]


@_rms_silu.register_fake
def _(x, gamma):
    return torch.empty_like(x)


@torch.library.custom_op("anima_flat::add_rms_silu", mutates_args=())
def _add_rms_silu(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(x)
    return _launch_fwd(x.contiguous(), residual.contiguous(), gamma, False)


@_add_rms_silu.register_fake
def _(x, residual, gamma):
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op("anima_flat::shuffle_rms_silu", mutates_args=())
def _shuffle_rms_silu(
    u: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(u)
    return _launch_fwd(u.contiguous(), None, gamma, True)


@_shuffle_rms_silu.register_fake
def _(u, gamma):
    B, h, w, _ = u.shape
    shape = (B, 2 * h, 2 * w, gamma.numel())
    return u.new_empty(shape), u.new_empty(shape)


@torch.library.custom_op("anima_flat::rms_silu_bwd", mutates_args=())
def _rms_silu_bwd(
    s: torch.Tensor,
    gamma: torch.Tensor,
    dy: torch.Tensor,
    gs: torch.Tensor | None,
    shuffled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    C = gamma.numel()
    s, dy = s.contiguous(), dy.contiguous()
    gs = gs.contiguous() if gs is not None else None
    N = s.numel() // C
    if shuffled:
        B, H, W, _ = s.shape
        ds_shape = (B, H // 2, W // 2, 4 * C)
    else:
        H = W = 0
        ds_shape = s.shape

    def launch(rows):
        nchunks = triton.cdiv(N, rows)
        ds = torch.empty(ds_shape, dtype=s.dtype, device=s.device)
        part = torch.empty(nchunks, C, dtype=torch.float32, device=s.device)
        _bwd_kernel[(nchunks,)](
            s,
            gamma,
            dy,
            gs if gs is not None else dy,
            ds,
            part,
            N,
            H,
            W,
            C=C,
            BLOCK=triton.next_power_of_2(C),
            ROWS=rows,
            HAS_GS=gs is not None,
            SHUF=shuffled,
        )
        return ds, part

    rows = tuned_rows(("rms_silu", C, N, s.dtype, gs is not None, shuffled), launch)
    ds, part = launch(rows)
    return ds, part.sum(0).to(gamma.dtype)


@_rms_silu_bwd.register_fake
def _(s, gamma, dy, gs, shuffled):
    if shuffled:
        B, H, W, _ = s.shape
        return s.new_empty((B, H // 2, W // 2, 4 * gamma.numel())), torch.empty_like(
            gamma
        )
    return torch.empty_like(s), torch.empty_like(gamma)


def _setup_rms_silu(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _bwd_rms_silu(ctx, dy):
    x, gamma = ctx.saved_tensors
    ds, dg = _rms_silu_bwd(x, gamma, dy, None, False)
    return ds, dg


_rms_silu.register_autograd(_bwd_rms_silu, setup_context=_setup_rms_silu)


def _setup_add(ctx, inputs, output):
    ctx.save_for_backward(output[0], inputs[2])


def _bwd_add(ctx, gs, dy):
    s, gamma = ctx.saved_tensors
    ds, dg = _rms_silu_bwd(s, gamma, dy, gs, False)
    return ds, ds, dg


_add_rms_silu.register_autograd(_bwd_add, setup_context=_setup_add)


def _setup_shuffle(ctx, inputs, output):
    ctx.save_for_backward(output[0], inputs[1])


def _bwd_shuffle(ctx, gs, dy):
    s, gamma = ctx.saved_tensors
    return _rms_silu_bwd(s, gamma, dy, gs, True)


_shuffle_rms_silu.register_autograd(_bwd_shuffle, setup_context=_setup_shuffle)


def rms_silu(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Channel RMS norm followed by SiLU.

    Computes `silu(normalize(x, dim=-1) * sqrt(C) * gamma)` in float32 and rounds once.

    Args:
        x: Activations with channels last, shape `[..., C]`, CUDA.
        gamma: Per-channel scale, shape `[C]`.

    Returns:
        Tensor with the shape and dtype of `x`.
    """
    return _rms_silu(x, gamma)


def add_rms_silu(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual add followed by `rms_silu` of the sum.

    Args:
        x: Branch output with channels last, shape `[..., C]`, CUDA.
        residual: Skip connection, same shape as `x`.
        gamma: Per-channel scale of the following norm, shape `[C]`.

    Returns:
        `(x + residual, rms_silu(x + residual, gamma))`.
    """
    return _add_rms_silu(x, residual, gamma)


def shuffle_rms_silu(
    u: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth-to-space by 2 followed by `rms_silu`.

    Args:
        u: Sub-pixel activations, shape `[B, H, W, 4 * C]`, CUDA. Channel `(2 * a + b) * C + c`
            is the value of channel `c` at output row `2 * i + a`, column `2 * j + b`.
        gamma: Per-channel scale, shape `[C]`.

    Returns:
        `(x, rms_silu(x, gamma))` with `x` of shape `[B, 2 * H, 2 * W, C]`.
    """
    return _shuffle_rms_silu(u, gamma)
