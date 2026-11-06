import torch
import triton
from triton import language as tl

from .common import _check

_HEAD_CONFIGS = [
    triton.Config({"HPP": hpp}, num_warps=nw) for hpp in (2, 4, 8, 16) for nw in (4, 8)
]


@triton.jit
def _section(
    src_ptr,
    dst_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    token,
    position,
    group,
    eps,
    SRC_ROW: tl.constexpr,
    SRC_COL: tl.constexpr,
    DST_ROW: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HPP: tl.constexpr,
    HALF: tl.constexpr,
    NORM: tl.constexpr,
    ROTATE: tl.constexpr,
):
    head = group * HPP + tl.arange(0, HPP)[:, None]
    dim = tl.arange(0, HALF)[None, :]
    mask = head < NUM_HEADS
    column = head * (2 * HALF) + dim
    src = src_ptr + token * SRC_ROW + SRC_COL + column
    dst = dst_ptr + token * DST_ROW + column

    a = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(src + HALF, mask=mask, other=0.0).to(tl.float32)
    if NORM:
        rstd = tl.rsqrt(
            (tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)) / (2 * HALF) + eps
        )[:, None]
        wa = tl.load(w_ptr + dim).to(tl.float32)
        wb = tl.load(w_ptr + HALF + dim).to(tl.float32)
        a = (a * rstd * wa).to(dst_ptr.dtype.element_ty).to(tl.float32)
        b = (b * rstd * wb).to(dst_ptr.dtype.element_ty).to(tl.float32)
    if ROTATE:
        cos = tl.load(cos_ptr + position * HALF + dim).to(tl.float32)
        sin = tl.load(sin_ptr + position * HALF + dim).to(tl.float32)
        out_a = a * cos - b * sin
        out_b = b * cos + a * sin
    else:
        out_a = a
        out_b = b
    tl.store(dst, out_a.to(dst_ptr.dtype.element_ty), mask=mask)
    tl.store(dst + HALF, out_b.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_HEAD_CONFIGS, key=["HEADS", "HALF"])
@triton.jit
def _qkv_norm_rope_kernel(
    qkv_ptr,
    q_ptr,
    k_ptr,
    qw_ptr,
    kw_ptr,
    cos_ptr,
    sin_ptr,
    S,
    eps,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    HPP: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    position = token % S
    width: tl.constexpr = HEADS * 2 * HALF
    row: tl.constexpr = 3 * width
    gq: tl.constexpr = (HEADS + HPP - 1) // HPP
    if group < gq:
        _section(
            qkv_ptr,
            q_ptr,
            qw_ptr,
            cos_ptr,
            sin_ptr,
            token,
            position,
            group,
            eps,
            row,
            tl.constexpr(0),
            width,
            HEADS,
            HPP,
            HALF,
            tl.constexpr(True),
            tl.constexpr(True),
        )
    else:
        _section(
            qkv_ptr,
            k_ptr,
            kw_ptr,
            cos_ptr,
            sin_ptr,
            token,
            position,
            group - gq,
            eps,
            row,
            width,
            width,
            HEADS,
            HPP,
            HALF,
            tl.constexpr(True),
            tl.constexpr(True),
        )


@triton.autotune(configs=_HEAD_CONFIGS, key=["HEADS", "HALF"])
@triton.jit
def _head_norm_kernel(
    x_ptr, y_ptr, w_ptr, eps, HEADS: tl.constexpr, HALF: tl.constexpr, HPP: tl.constexpr
):
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    width: tl.constexpr = HEADS * 2 * HALF
    _section(
        x_ptr,
        y_ptr,
        w_ptr,
        w_ptr,
        w_ptr,
        token,
        token,
        group,
        eps,
        width,
        tl.constexpr(0),
        width,
        HEADS,
        HPP,
        HALF,
        tl.constexpr(True),
        tl.constexpr(False),
    )


@torch.library.custom_op("anima_flat::qkv_norm_rope", mutates_args=())
def _qkv_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, _ = qkv.shape
    half = cos.shape[1]
    q = qkv.new_empty((B, S, heads, 2 * half))
    k = torch.empty_like(q)
    grid = lambda meta: (B * S, 2 * triton.cdiv(heads, meta["HPP"]))
    _qkv_norm_rope_kernel[grid](
        qkv, q, k, q_weight, k_weight, cos, sin, S, eps, HEADS=heads, HALF=half
    )
    return q, k


@_qkv_norm_rope.register_fake
def _(qkv, q_weight, k_weight, cos, sin, heads, eps):
    B, S, _ = qkv.shape
    q = qkv.new_empty((B, S, heads, 2 * cos.shape[1]))
    return q, torch.empty_like(q)


@torch.library.custom_op("anima_flat::head_norm", mutates_args=())
def _head_norm(
    x: torch.Tensor, weight: torch.Tensor, heads: int, eps: float
) -> torch.Tensor:
    B, S, _ = x.shape
    hd = weight.numel()
    y = x.new_empty((B, S, heads, hd))
    grid = lambda meta: (B * S, triton.cdiv(heads, meta["HPP"]))
    _head_norm_kernel[grid](x, y, weight, eps, HEADS=heads, HALF=hd // 2)
    return y


@_head_norm.register_fake
def _(x, weight, heads, eps):
    return x.new_empty((x.shape[0], x.shape[1], heads, weight.numel()))


def qkv_norm_rope(qkv, q_weight, k_weight, cos, sin, heads, eps=1e-5):
    _check(qkv, q_weight, k_weight, cos, sin)
    if (
        not qkv.is_contiguous()
        or qkv.ndim != 3
        or qkv.shape[-1] != 3 * heads * 2 * cos.shape[1]
    ):
        raise ValueError("qkv must be contiguous [B, S, 3 * heads * head_dim]")
    if (
        cos.shape != sin.shape
        or cos.shape[0] != qkv.shape[1]
        or (cos.shape[1] & (cos.shape[1] - 1))
    ):
        raise ValueError("cos/sin must be [S, head_dim // 2] with a power-of-two width")
    q, k = _qkv_norm_rope(
        qkv,
        q_weight.contiguous(),
        k_weight.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        heads,
        eps,
    )
    # V needs no norm or rotation: return a strided view of qkv instead of copying it.
    B, S, _ = qkv.shape
    v = qkv.view(B, S, 3, heads, 2 * cos.shape[1])[:, :, 2]
    return q, k, v


def head_norm(x, weight, heads, eps=1e-5):
    _check(x, weight)
    if not x.is_contiguous() or x.shape[-1] != heads * weight.numel():
        raise ValueError("x must be contiguous [B, S, heads * head_dim]")
    return _head_norm(x, weight.contiguous(), heads, eps)
