import torch
import triton
from triton import language as tl

from .common import _check


@triton.jit
def _ctx_scale_kernel(
    out_ptr,
    lse_ptr,
    ln_ptr,
    y_ptr,
    S,
    HEADS: tl.constexpr,
    HD: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    batch = token // S
    position = token % S
    heads = tl.arange(0, BLOCK_H)
    head_mask = heads < HEADS
    lse = tl.load(
        lse_ptr + (batch * HEADS + heads) * S + position, mask=head_mask, other=0.0
    )
    ln_pad = tl.load(ln_ptr + batch)
    gain = 1.0 / (1.0 + tl.exp(ln_pad - lse))
    dims = tl.arange(0, HD)
    off = token * (HEADS * HD) + heads[:, None] * HD + dims[None, :]
    x = tl.load(out_ptr + off, mask=head_mask[:, None], other=0.0).to(tl.float32)
    tl.store(
        y_ptr + off,
        (x * gain[:, None]).to(y_ptr.dtype.element_ty),
        mask=head_mask[:, None],
    )


@torch.library.custom_op("anima_flat::ctx_scale", mutates_args=())
def _ctx_scale(
    out: torch.Tensor, lse: torch.Tensor, ln_pad: torch.Tensor
) -> torch.Tensor:
    B, S, H, hd = out.shape
    y = torch.empty_like(out)
    _ctx_scale_kernel[(B * S,)](
        out,
        lse,
        ln_pad,
        y,
        S,
        HEADS=tl.constexpr(H),
        HD=tl.constexpr(hd),
        BLOCK_H=tl.constexpr(triton.next_power_of_2(H)),
    )
    return y


@_ctx_scale.register_fake
def _(out, lse, ln_pad):
    return torch.empty_like(out)


def ctx_scale(out, lse, ln_pad):
    _check(out, lse, ln_pad)
    if out.ndim != 4 or lse.shape != (out.shape[0], out.shape[2], out.shape[1]):
        raise ValueError("out must be [B, S, H, hd] and lse [B, H, S]")
    return _ctx_scale(
        out.contiguous(), lse.float().contiguous(), ln_pad.float().contiguous()
    )
