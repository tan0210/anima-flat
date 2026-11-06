import torch
import triton
from triton import language as tl

from .common import _check, tuned_rows

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
    v_ptr,
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
    elif group < 2 * gq:
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
    else:
        _section(
            qkv_ptr,
            v_ptr,
            kw_ptr,
            cos_ptr,
            sin_ptr,
            token,
            position,
            group - 2 * gq,
            eps,
            row,
            2 * width,
            width,
            HEADS,
            HPP,
            HALF,
            tl.constexpr(False),
            tl.constexpr(False),
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
        qkv, q, k, q, q_weight, k_weight, cos, sin, S, eps, HEADS=heads, HALF=half
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


_BWD_CONFIGS = [triton.Config({}, num_warps=nw) for nw in (4, 8)]


@triton.jit
def _section_bwd(
    src_ptr,
    dout_ptr,
    dsrc_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    token,
    position,
    group,
    eps,
    valid,
    SRC_ROW: tl.constexpr,
    SRC_COL: tl.constexpr,
    DOUT_ROW: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HPP: tl.constexpr,
    HALF: tl.constexpr,
    ROTATE: tl.constexpr,
):
    head = group * HPP + tl.arange(0, HPP)[:, None]
    dim = tl.arange(0, HALF)[None, :]
    mask = (head < NUM_HEADS) & (dim < tl.where(valid, HALF, 0))
    column = head * (2 * HALF) + dim
    src = src_ptr + token * SRC_ROW + SRC_COL + column
    dsrc = dsrc_ptr + token * SRC_ROW + SRC_COL + column
    dout = dout_ptr + token * DOUT_ROW + column

    a = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(src + HALF, mask=mask, other=0.0).to(tl.float32)
    da = tl.load(dout, mask=mask, other=0.0).to(tl.float32)
    db = tl.load(dout + HALF, mask=mask, other=0.0).to(tl.float32)
    if ROTATE:
        cos = tl.load(cos_ptr + position * HALF + dim).to(tl.float32)
        sin = tl.load(sin_ptr + position * HALF + dim).to(tl.float32)
        da, db = da * cos + db * sin, db * cos - da * sin
    rstd = tl.rsqrt((tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)) / (2 * HALF) + eps)[
        :, None
    ]
    wa = tl.load(w_ptr + dim).to(tl.float32)
    wb = tl.load(w_ptr + HALF + dim).to(tl.float32)
    xa = a * rstd
    xb = b * rstd
    ua = da * wa
    ub = db * wb
    c = ((tl.sum(ua * xa, axis=1) + tl.sum(ub * xb, axis=1)) / (2 * HALF))[:, None]
    tl.store(dsrc, (rstd * (ua - xa * c)).to(dsrc_ptr.dtype.element_ty), mask=mask)
    tl.store(
        dsrc + HALF, (rstd * (ub - xb * c)).to(dsrc_ptr.dtype.element_ty), mask=mask
    )
    return tl.sum(da * xa, axis=0), tl.sum(db * xb, axis=0)


@triton.autotune(configs=_BWD_CONFIGS, key=["HEADS", "HALF", "ROWS"])
@triton.jit
def _qkv_norm_rope_bwd_kernel(
    qkv_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dqkv_ptr,
    qw_ptr,
    kw_ptr,
    cos_ptr,
    sin_ptr,
    pw_ptr,
    S,
    eps,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    HPP: tl.constexpr,
    ROWS: tl.constexpr,
):
    chunk = tl.program_id(0)
    group = tl.program_id(1)
    batch = chunk // ((S + ROWS - 1) // ROWS)
    first = (chunk % ((S + ROWS - 1) // ROWS)) * ROWS
    width: tl.constexpr = HEADS * 2 * HALF
    row: tl.constexpr = 3 * width
    gq: tl.constexpr = (HEADS + HPP - 1) // HPP
    if group >= 2 * gq:
        for r in range(ROWS):
            s = first + r
            if s < S:
                _section(
                    dv_ptr,
                    dqkv_ptr + 2 * width,
                    qw_ptr,
                    cos_ptr,
                    sin_ptr,
                    (batch * S + s).to(tl.int64),
                    s,
                    group - 2 * gq,
                    eps,
                    width,
                    tl.constexpr(0),
                    row,
                    HEADS,
                    HPP,
                    HALF,
                    tl.constexpr(False),
                    tl.constexpr(False),
                )
    else:
        is_q = group < gq
        local = tl.where(is_q, group, group - gq)
        acc_a = tl.zeros([HALF], dtype=tl.float32)
        acc_b = tl.zeros([HALF], dtype=tl.float32)
        for r in range(ROWS):
            s = first + r
            token = (batch * S + s).to(tl.int64)
            position = tl.minimum(s, S - 1)
            if is_q:
                wa, wb = _section_bwd(
                    qkv_ptr,
                    dq_ptr,
                    dqkv_ptr,
                    qw_ptr,
                    cos_ptr,
                    sin_ptr,
                    token,
                    position,
                    local,
                    eps,
                    s < S,
                    row,
                    tl.constexpr(0),
                    width,
                    HEADS,
                    HPP,
                    HALF,
                    tl.constexpr(True),
                )
            else:
                wa, wb = _section_bwd(
                    qkv_ptr,
                    dk_ptr,
                    dqkv_ptr,
                    kw_ptr,
                    cos_ptr,
                    sin_ptr,
                    token,
                    position,
                    local,
                    eps,
                    s < S,
                    row,
                    width,
                    width,
                    HEADS,
                    HPP,
                    HALF,
                    tl.constexpr(True),
                )
            acc_a += wa
            acc_b += wb
        dim = tl.arange(0, HALF)
        out = pw_ptr + (chunk * tl.num_programs(1) + group) * (2 * HALF)
        tl.store(out + dim, acc_a)
        tl.store(out + HALF + dim, acc_b)


@triton.autotune(configs=_BWD_CONFIGS, key=["HEADS", "HALF", "ROWS"])
@triton.jit
def _head_norm_bwd_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    w_ptr,
    pw_ptr,
    S,
    eps,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    HPP: tl.constexpr,
    ROWS: tl.constexpr,
):
    chunk = tl.program_id(0)
    group = tl.program_id(1)
    batch = chunk // ((S + ROWS - 1) // ROWS)
    first = (chunk % ((S + ROWS - 1) // ROWS)) * ROWS
    width: tl.constexpr = HEADS * 2 * HALF
    acc_a = tl.zeros([HALF], dtype=tl.float32)
    acc_b = tl.zeros([HALF], dtype=tl.float32)
    for r in range(ROWS):
        s = first + r
        token = (batch * S + s).to(tl.int64)
        wa, wb = _section_bwd(
            x_ptr,
            dy_ptr,
            dx_ptr,
            w_ptr,
            w_ptr,
            w_ptr,
            token,
            token,
            group,
            eps,
            s < S,
            width,
            tl.constexpr(0),
            width,
            HEADS,
            HPP,
            HALF,
            tl.constexpr(False),
        )
        acc_a += wa
        acc_b += wb
    dim = tl.arange(0, HALF)
    out = pw_ptr + (chunk * tl.num_programs(1) + group) * (2 * HALF)
    tl.store(out + dim, acc_a)
    tl.store(out + HALF + dim, acc_b)


def _bwd_hpp(heads):
    return min(4, triton.next_power_of_2(heads))


def _head_norm_bwd(x, dy, weight, heads, eps):
    B, S, _ = x.shape
    hd = weight.numel()
    hpp = _bwd_hpp(heads)
    groups = triton.cdiv(heads, hpp)
    dy = dy.contiguous()

    def launch(rows):
        chunks = B * triton.cdiv(S, rows)
        dx = torch.empty_like(x)
        pw = torch.empty(chunks, groups, hd, dtype=torch.float32, device=x.device)
        _head_norm_bwd_kernel[(chunks, groups)](
            x,
            dy,
            dx,
            weight,
            pw,
            S,
            eps,
            HEADS=heads,
            HALF=hd // 2,
            HPP=hpp,
            ROWS=rows,
        )
        return dx, pw

    rows = tuned_rows(("head_norm_bwd", B, S, heads, hd, x.dtype), launch)
    dx, pw = launch(rows)
    return dx, pw.sum((0, 1)).to(weight.dtype)


def _head_norm_setup(ctx, inputs, output):
    x, weight, heads, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.heads = heads
    ctx.eps = eps


def _head_norm_backward(ctx, dy):
    x, weight = ctx.saved_tensors
    dx, dw = _head_norm_bwd(x, dy, weight, ctx.heads, ctx.eps)
    return dx, dw, None, None


torch.library.register_autograd(
    "anima_flat::head_norm", _head_norm_backward, setup_context=_head_norm_setup
)


@torch.library.custom_op("anima_flat::qkv_norm_rope_v", mutates_args=())
def _qkv_norm_rope_v(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, _ = qkv.shape
    half = cos.shape[1]
    q = qkv.new_empty((B, S, heads, 2 * half))
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    grid = lambda meta: (B * S, 3 * triton.cdiv(heads, meta["HPP"]))
    _qkv_norm_rope_kernel[grid](
        qkv, q, k, v, q_weight, k_weight, cos, sin, S, eps, HEADS=heads, HALF=half
    )
    return q, k, v


@_qkv_norm_rope_v.register_fake
def _(qkv, q_weight, k_weight, cos, sin, heads, eps):
    B, S, _ = qkv.shape
    q = qkv.new_empty((B, S, heads, 2 * cos.shape[1]))
    return q, torch.empty_like(q), torch.empty_like(q)


def _qkv_norm_rope_setup(ctx, inputs, output):
    qkv, q_weight, k_weight, cos, sin, heads, eps = inputs
    ctx.save_for_backward(qkv, q_weight, k_weight, cos, sin)
    ctx.heads = heads
    ctx.eps = eps


def _qkv_norm_rope_backward(ctx, dq, dk, dv):
    qkv, q_weight, k_weight, cos, sin = ctx.saved_tensors
    heads, eps = ctx.heads, ctx.eps
    B, S, _ = qkv.shape
    half = cos.shape[1]
    hpp = _bwd_hpp(heads)
    gq = triton.cdiv(heads, hpp)
    dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()

    def launch(rows):
        chunks = B * triton.cdiv(S, rows)
        dqkv = torch.empty_like(qkv)
        pw = torch.empty(
            chunks, 3 * gq, 2 * half, dtype=torch.float32, device=qkv.device
        )
        _qkv_norm_rope_bwd_kernel[(chunks, 3 * gq)](
            qkv,
            dq,
            dk,
            dv,
            dqkv,
            q_weight,
            k_weight,
            cos,
            sin,
            pw,
            S,
            eps,
            HEADS=heads,
            HALF=half,
            HPP=hpp,
            ROWS=rows,
        )
        return dqkv, pw

    rows = tuned_rows(("qkv_norm_rope_bwd", B, S, heads, half, qkv.dtype), launch)
    dqkv, pw = launch(rows)
    dqw = pw[:, :gq].sum((0, 1)).to(q_weight.dtype)
    dkw = pw[:, gq : 2 * gq].sum((0, 1)).to(k_weight.dtype)
    return dqkv, dqw, dkw, None, None, None, None


torch.library.register_autograd(
    "anima_flat::qkv_norm_rope_v",
    _qkv_norm_rope_backward,
    setup_context=_qkv_norm_rope_setup,
)


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
    args = (
        qkv,
        q_weight.contiguous(),
        k_weight.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        heads,
        eps,
    )
    if torch.is_grad_enabled() and (qkv.requires_grad or q_weight.requires_grad):
        return _qkv_norm_rope_v(*args)
    q, k = _qkv_norm_rope(*args)
    # V needs no norm or rotation: return a strided view of qkv instead of copying it.
    B, S, _ = qkv.shape
    v = qkv.view(B, S, 3, heads, 2 * cos.shape[1])[:, :, 2]
    return q, k, v


def head_norm(x, weight, heads, eps=1e-5):
    _check(x, weight)
    if not x.is_contiguous() or x.shape[-1] != heads * weight.numel():
        raise ValueError("x must be contiguous [B, S, heads * head_dim]")
    return _head_norm(x, weight.contiguous(), heads, eps)
