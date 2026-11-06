import functools
import os

import torch
import torch.nn.functional as F

from .kernels.ctx_scale import ctx_scale

try:
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd  # type: ignore
except Exception:
    _flash_attn_bwd = None
    _flash_attn_fwd = None


@functools.cache
def _capable(index):
    return torch.cuda.get_device_capability(index)[0] >= 8


def use_fa4(t: torch.Tensor) -> bool:
    """Decide whether FlashAttention-4 can be used for tensors like `t`.

    The `ANIMA_FLAT_ATTENTION=sdpa` environment variable forces the SDPA fallback.

    Args:
        t: Representative attention input; its device and dtype are checked.

    Returns:
        True if FA4 is importable, `t` is a bf16/fp16 CUDA tensor, and the GPU has
        compute capability >= 8; False otherwise.
    """
    if os.environ.get("ANIMA_FLAT_ATTENTION", "fa4") == "sdpa":
        return False
    return (
        _flash_attn_fwd is not None
        and t.is_cuda
        and t.dtype in (torch.bfloat16, torch.float16)
        and _capable(t.device.index)
    )


@torch.library.custom_op("anima_flat::fa_fwd", mutates_args=())
def _fa_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_ = _flash_attn_fwd(q, k, v, causal=False, return_lse=True)  # type: ignore
    expected = (q.shape[0], q.shape[2], q.shape[1])
    if tuple(lse.shape) != expected:
        raise RuntimeError(
            f"Unexpected FA4 lse shape {tuple(lse.shape)}, expected {expected}"
        )
    return out, lse


@_fa_fwd.register_fake
def _(q, k, v):
    return torch.empty_like(q), q.new_empty(
        (q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32
    )


@torch.library.custom_op("anima_flat::fa_fwd_nolse", mutates_args=())
def _fa_fwd_nolse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out, *_ = _flash_attn_fwd(q, k, v, causal=False, return_lse=False)  # type: ignore
    return out


@_fa_fwd_nolse.register_fake
def _(q, k, v):
    return q.new_empty(q.shape)


@torch.library.custom_op("anima_flat::fa_fwd_k", mutates_args=())
def _fa_fwd_k(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqused_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_ = _flash_attn_fwd(  # type: ignore
        q, k, v, seqused_k=seqused_k, causal=False, return_lse=True
    )
    return out, lse


@_fa_fwd_k.register_fake
def _(q, k, v, seqused_k):
    return torch.empty_like(q), q.new_empty(
        (q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32
    )


@torch.library.custom_op("anima_flat::fa_bwd", mutates_args=())
def _fa_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv, *_ = _flash_attn_bwd(  # type: ignore
        q, k, v, out, dout.contiguous(), lse, causal=False, deterministic=False
    )
    return dq, dk, dv


@_fa_bwd.register_fake
def _(dout, q, k, v, out, lse):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v = inputs
    out, lse = output
    ctx.mark_non_differentiable(lse)
    ctx.save_for_backward(q, k, v, out, lse)


def _backward(ctx, dout, dlse):
    q, k, v, out, lse = ctx.saved_tensors
    return _fa_bwd(dout, q, k, v, out, lse)


_fa_fwd.register_autograd(_backward, setup_context=_setup_context)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full (non-causal) multi-head attention.

    Uses FlashAttention-4 when `use_fa4(q)` holds (differentiable), otherwise PyTorch SDPA.
    Inputs are not copied: strided views are fine as long as the last dim has stride 1 (FA4
    also wants 16-byte-aligned pointers and strides that are multiples of 8 elements, else
    it copies internally). The softmax LSE is only computed when autograd needs it.

    Args:
        q: Queries, shape `[B, Sq, H, hd]`.
        k: Keys, shape `[B, Sk, H, hd]`.
        v: Values, shape `[B, Sk, H, hd]`.

    Returns:
        Attention output with heads merged, shape `[B, Sq, H * hd]`, dtype of `q`.
    """
    if use_fa4(q):
        if torch.is_grad_enabled() and (
            q.requires_grad or k.requires_grad or v.requires_grad
        ):
            out, _ = _fa_fwd(q, k, v)
        else:
            out = _fa_fwd_nolse(q, k, v)
        return out.flatten(2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).flatten(2)


def cross_attention_trimmed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    ln_pad: torch.Tensor,
) -> torch.Tensor:
    """Cross-attention over a trimmed key/value context, corrected for the removed padding.

    Requires FA4. Only the first `seqused_k[b]` keys of batch item `b` are attended; the
    output is rescaled so it matches attention over the full zero-padded context.

    Args:
        q: Queries, shape `[B, Sq, H, hd]`.
        k: Trimmed keys, shape `[B, Sk_trim, H, hd]`.
        v: Trimmed values, shape `[B, Sk_trim, H, hd]`.
        seqused_k: Valid key count per batch item, int32, shape `[B]`.
        ln_pad: `log(S_full - seqused_k)` per batch item (log of the number of padded keys),
            float32, shape `[B]`.

    Returns:
        Attention output with heads merged, shape `[B, Sq, H * hd]`.
    """
    out, lse = _fa_fwd_k(q.contiguous(), k, v, seqused_k)
    return ctx_scale(out, lse, ln_pad).flatten(2)
