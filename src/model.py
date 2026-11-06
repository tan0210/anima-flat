import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import attention, cross_attention_trimmed, use_fa4
from .config import FORMAT, FlatConfig
from .kernels.adaln import adaln, residual_adaln
from .kernels.fp8 import (
    FP8Weight,
    adaln_fp8,
    fp8_linear,
    quantize_weight,
    residual_adaln_fp8,
    rowquant,
    supported,
)
from .kernels.gemm import linear_gelu, mlp
from .kernels.rope import head_norm, qkv_norm_rope
from .lora import has_lora, merge_lora_file
from .sample import SAMPLERS, er_sde, euler, sigmas

NORM_EPS = 1e-6
MODULATION_KEYS = ("t_lin1.", "t_lin2.", "t_norm", "mod_w1", "mod_w2", "out_w2")


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (cos half first, then sin half).

    Args:
        t: Timesteps, shape `[B]`.
        dim: Embedding width (even).

    Returns:
        Float32 embedding, shape `[B, dim]`.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def rope_tables(
    h: int,
    w: int,
    head_dim: int,
    device: str | torch.device,
    rope_scale: tuple = (1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 3D (time, height, width) rotary tables for an `h x w` token grid.

    Args:
        h: Grid height in tokens.
        w: Grid width in tokens.
        head_dim: Attention head width.
        device: Device to build the tables on.
        rope_scale: NTK scaling of the (height, width) frequencies.

    Returns:
        `(cos, sin)`, each float32 of shape `[h * w, head_dim // 2]`.
    """
    dh = head_dim // 6 * 2
    dt = head_dim - 2 * dh
    ntk = dh / (dh - 2)
    ar = torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh
    inv_h = 1.0 / ((10000.0 * rope_scale[0] ** ntk) ** ar)
    inv_w = 1.0 / ((10000.0 * rope_scale[1] ** ntk) ** ar)
    ah = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)[
        :, None, :
    ].expand(h, w, -1)
    aw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)[
        None, :, :
    ].expand(h, w, -1)
    at = torch.zeros(h, w, dt // 2, device=device)
    ang = torch.cat([at, ah, aw], dim=-1).reshape(h * w, -1)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings (half-split convention) in float32.

    Args:
        x: Input, shape `[B, S, H, hd]`.
        cos: Cosine table, shape `[S, hd // 2]`.
        sin: Sine table, shape `[S, hd // 2]`.

    Returns:
        Rotated tensor with the shape and dtype of `x`.
    """
    x1, x2 = x.float().chunk(2, dim=-1)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


class Context:
    """Precomputed per-layer cross-attention keys/values for one text context.

    Attributes:
        kv: Per-layer `(k, v)` pairs, each `[B, S, H, hd]`.
        seqused: Valid context length per batch item (int32, `[B]`) if trimmed, else None.
        ln_pad: `log` of the number of trimmed padding keys per batch item (`[B]`) if trimmed, else None.
    """

    def __init__(
        self,
        kv: list[tuple[torch.Tensor, torch.Tensor]],
        seqused: torch.Tensor | None = None,
        ln_pad: torch.Tensor | None = None,
    ) -> None:
        self.kv = kv
        self.seqused = seqused
        self.ln_pad = ln_pad

    def attend(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        """Cross-attend queries to this context at one layer.

        Args:
            layer: Block index.
            q: Normalized queries, shape `[B, Sq, H, hd]`.

        Returns:
            Attention output, shape `[B, Sq, H * hd]`.
        """
        k, v = self.kv[layer]
        if self.seqused is None:
            return attention(q, k, v)
        return cross_attention_trimmed(q, k, v, self.seqused, self.ln_pad)  # type: ignore


class Block(nn.Module):
    """One transformer block: self-attention, cross-attention and MLP, each adaLN-modulated.

    Args:
        c: Model config.
    """

    def __init__(self, c: FlatConfig) -> None:
        super().__init__()
        D, hd = c.dim, c.head_dim
        self.heads, self.hd, self.eps = c.heads, hd, c.qk_eps
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.q_norm = nn.Parameter(torch.ones(hd))
        self.k_norm = nn.Parameter(torch.ones(hd))
        self.o1 = nn.Linear(D, D, bias=False)
        self.xq = nn.Linear(D, D, bias=False)
        self.xkv = nn.Linear(c.text_dim, 2 * D, bias=False)
        self.xq_norm = nn.Parameter(torch.ones(hd))
        self.xk_norm = nn.Parameter(torch.ones(hd))
        self.o2 = nn.Linear(D, D, bias=False)
        self.up = nn.Linear(D, c.mlp_dim, bias=False)
        self.down = nn.Linear(c.mlp_dim, D, bias=False)

    def modulate(
        self, x: torch.Tensor, m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameter-free LayerNorm followed by shift/scale modulation.

        Args:
            x: Tokens, shape `[B, S, D]`.
            m: Modulation for one sub-layer, shape `[B, 3, D]` ordered (shift, scale + 1, gate).

        Returns:
            `(y, gate)`: modulated tokens `[B, S, D]` and the gate `[B, 1, D]`.
        """
        shift, scale, gate = (t[:, None] for t in m.unbind(1))
        return F.layer_norm(x, x.shape[-1:], eps=NORM_EPS) * scale + shift, gate

    def forward(
        self,
        x: torch.Tensor,
        ck: torch.Tensor,
        cv: torch.Tensor,
        m: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        trim: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Reference block forward.

        Args:
            x: Tokens, shape `[B, S, D]`.
            ck: This block's cross-attention keys, shape `[B, Sc, H, hd]`.
            cv: This block's cross-attention values, shape `[B, Sc, H, hd]`.
            m: Modulation for this block, shape `[B, 3, 3, D]`.
            cos: Rotary cosines, shape `[S, hd // 2]`.
            sin: Rotary sines, shape `[S, hd // 2]`.
            trim: `(seqused, ln_pad)` from a trimmed `Context`, or None for full cross-attention.

        Returns:
            Updated tokens, shape `[B, S, D]`.
        """
        B, S, _ = x.shape  # B, S, D
        y, g = self.modulate(x, m[:, 0])
        q, k, v = self.qkv(y).view(B, S, 3, self.heads, self.hd).unbind(2)
        q = apply_rope(
            F.rms_norm(q, (self.hd,), self.q_norm.to(q.dtype), self.eps), cos, sin
        )
        k = apply_rope(
            F.rms_norm(k, (self.hd,), self.k_norm.to(k.dtype), self.eps), cos, sin
        )
        x = x + g * self.o1(attention(q, k, v))
        y, g = self.modulate(x, m[:, 1])
        q = self.xq(y).view(B, S, self.heads, self.hd)
        q = F.rms_norm(q, (self.hd,), self.xq_norm.to(q.dtype), self.eps)
        if trim is None:
            c = attention(q, ck, cv)
        else:
            c = cross_attention_trimmed(q, ck, cv, *trim)
        x = x + g * self.o2(c)
        y, g = self.modulate(x, m[:, 2])
        # Adapters wrapping `up` or `down` must run inside the MLP, so only plain layers fuse.
        if type(self.up) is nn.Linear:
            if type(self.down) is nn.Linear:
                return x + g * mlp(y, self.up.weight, self.down.weight)
            h = linear_gelu(y, self.up.weight)
        else:
            h = F.gelu(self.up(y))
        return x + g * self.down(h)


FP8_LAYERS = ("qkv", "o1", "xq", "o2", "up", "down")


class AnimaFlat(nn.Module):
    """Anima diffusion transformer with fused/flattened weights.

    Attributes:
        fused: Use the fused Triton forward when running on CUDA without autograd.
        trim_context: Trim zero-padded text context in `encode_context` (requires FA4; ignored
            while autograd is enabled).
        checkpointing: Recompute each block during backward instead of storing its activations.
            Only matters while autograd is enabled.
        compile_blocks: Run the blocks of `forward_reference` through `torch.compile`.

    Args:
        cfg: Model config.
    """

    def __init__(self, cfg: FlatConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D, L, R = cfg.dim, cfg.layers, cfg.lora_dim
        self.fused = True
        self.trim_context = False
        self.checkpointing = True
        self.compile_blocks = False
        self._compiled = {}
        self.patch = nn.Linear(4 * cfg.in_ch, D, bias=False)
        self.t_lin1 = nn.Linear(D, D, bias=False)
        self.t_lin2 = nn.Linear(D, 3 * D, bias=False)
        self.t_norm = nn.Parameter(torch.ones(D))
        self.mod_w1 = nn.Parameter(torch.zeros((3 * L + 1) * R, D))
        self.mod_w2 = nn.Parameter(torch.zeros(3 * L, 3 * D, R))
        self.out_w2 = nn.Parameter(torch.zeros(2 * D, R))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(L))
        self.out_proj = nn.Linear(D, 4 * cfg.in_ch, bias=False)
        self._tables = {}
        self.fp8: list[dict[str, FP8Weight]] | None = None
        self.fp8_fast_accum = False

    def tables(
        self, h: int, w: int, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get (and cache) rotary tables for a token grid.

        Args:
            h: Grid height in tokens (latent height // 2).
            w: Grid width in tokens (latent width // 2).
            device: Target device.

        Returns:
            `(cos, sin)`, each `[h * w, head_dim // 2]`.
        """
        key = (h, w, device)
        if key not in self._tables:
            self._tables[key] = rope_tables(
                h, w, self.cfg.head_dim, device, self.cfg.rope_scale
            )
        return self._tables[key]

    def modulation(
        self, t: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute all adaLN modulation parameters from the timestep.

        Args:
            t: Timesteps (sigmas), shape `[B]`.
            dtype: Output dtype.

        Returns:
            `(mod, fin)`: `mod` of shape `[B, L, 3, 3, D]` indexed by (layer, sub-layer, shift/scale+1/gate),
            and the final-layer (shift, scale) `fin` of shape `[B, 2 * D]`.
        """
        c = self.cfg
        B, L, D, R = t.shape[0], c.layers, c.dim, c.lora_dim
        ts = timestep_embedding(t, D).to(self.mod_w1.dtype)
        temb = self.t_lin2(F.silu(self.t_lin1(ts)))
        emb = F.rms_norm(ts, (D,), self.t_norm, NORM_EPS)
        h = (F.silu(emb) @ self.mod_w1.T).view(B, 3 * L + 1, R)
        blk = (
            torch.bmm(h[:, :-1].transpose(0, 1), self.mod_w2.transpose(1, 2)).transpose(
                0, 1
            )
            + temb[:, None]
        )
        shift, scale, gate = blk.view(B, L, 3, 3, D).unbind(3)
        mod = torch.stack((shift, scale + 1, gate), dim=3).to(dtype)
        fin = (h[:, -1] @ self.out_w2.T + temb[:, : 2 * D]).to(dtype)
        return mod, fin

    def encode_context(self, ctx: torch.Tensor) -> Context:
        """Precompute cross-attention keys/values of every layer for a text context.

        Call once per prompt batch and reuse across denoising steps. If `trim_context` is set
        and FA4 is usable, trailing all-zero padding is dropped (not while autograd is enabled).

        Args:
            ctx: Text embeddings, shape `[B, S, text_dim]`, zero-padded along `S`.

        Returns:
            A `Context` holding per-layer keys/values.
        """
        c = self.cfg
        B, S, _ = ctx.shape
        seqused = ln_pad = None
        if self.trim_context and not torch.is_grad_enabled() and use_fa4(ctx):
            index = torch.arange(1, S + 1, device=ctx.device)
            lengths = ((ctx != 0).any(-1) * index).amax(1).clamp_min(1)
            keep = min(S, (int(lengths.max()) + 7) // 8 * 8)
            ctx = ctx[:, :keep]
            seqused = lengths.to(torch.int32).contiguous()
            ln_pad = torch.log((S - lengths).float())
            S = keep
        kv = []
        for b in self.blocks:
            k, v = b.xkv(ctx).view(B, S, 2, c.heads, c.head_dim).unbind(2)  # type: ignore
            k = F.rms_norm(k, (c.head_dim,), b.xk_norm.to(k.dtype), c.qk_eps)  # type: ignore
            kv.append((k.contiguous(), v.contiguous()))
        return Context(kv, seqused, ln_pad)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Split a latent into 2x2 patches and embed them.

        Args:
            x: Latent, shape `[B, H, W, C]` (H and W even).

        Returns:
            Token embeddings, shape `[B, (H // 2) * (W // 2), D]`.
        """
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5)
        return self.patch(x.reshape(B, (H // 2) * (W // 2), 4 * C))

    def unpatchify(self, y: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Inverse of the patch layout: fold patch predictions back into a latent.

        Args:
            y: Per-patch outputs, shape `[B, (H // 2) * (W // 2), 4 * C]`.
            H: Latent height.
            W: Latent width.

        Returns:
            Latent, shape `[B, H, W, C]`.
        """
        B, C = y.shape[0], self.cfg.in_ch
        y = y.reshape(B, H // 2, W // 2, 2, 2, C).permute(0, 1, 3, 2, 4, 5)
        return y.reshape(B, H, W, C)

    def _block(self, i: int) -> nn.Module:
        b = self.blocks[i]
        if not self.compile_blocks:
            return b
        if i not in self._compiled:
            self._compiled[i] = torch.compile(b)
        return self._compiled[i]

    def forward_reference(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: Context,
        mods: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Unfused forward pass in plain PyTorch (differentiable).

        Args:
            x: Noisy latent, shape `[B, H, W, C]`.
            t: Timesteps (sigmas), shape `[B]`; ignored if `mods` is given.
            ctx: Context from `encode_context`.
            mods: Optional precomputed `(mod, fin)` as returned by `modulation`, with batch size `B`
                (broadcast views are fine).

        Returns:
            Predicted velocity, shape `[B, H, W, C]`.
        """
        _, H, W, _ = x.shape  # B, H, W, C
        cos, sin = self.tables(H // 2, W // 2, x.device)
        z = self.patchify(x)
        mod, fin = mods if mods is not None else self.modulation(t, z.dtype)
        trim = None if ctx.seqused is None else (ctx.seqused, ctx.ln_pad)
        for i in range(len(self.blocks)):
            b = self._block(i)
            k, v = ctx.kv[i]
            args = (z, k, v, mod[:, i], cos, sin, trim)
            if self.checkpointing and torch.is_grad_enabled():
                z = checkpoint(b, *args, use_reentrant=False)
            else:
                z = b(*args)
        shift, scale = fin.chunk(2, dim=-1)
        z = (
            F.layer_norm(z, z.shape[-1:], eps=NORM_EPS) * (1 + scale[:, None])
            + shift[:, None]
        )
        return self.unpatchify(self.out_proj(z), H, W)

    @torch.no_grad()
    def forward_fused(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: Context,
        mods: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Inference forward pass using the fused Triton kernels (CUDA only, no autograd).

        Args:
            x: Noisy latent, shape `[B, H, W, C]`, on CUDA.
            t: Timesteps (sigmas), shape `[B]`; ignored if `mods` is given.
            ctx: Context from `encode_context`.
            mods: Optional precomputed `(mod, fin)` as returned by `modulation`, with batch size `B`
                (broadcast views are fine).

        Returns:
            Predicted velocity, shape `[B, H, W, C]`.
        """
        c = self.cfg
        _, H, W, _ = x.shape  # B, H, W, C
        cos, sin = self.tables(H // 2, W // 2, x.device)
        z = self.patchify(x)
        mod, fin = mods if mods is not None else self.modulation(t, z.dtype)
        mod = mod.permute(1, 2, 3, 0, 4).contiguous()
        hid = adaln(z, mod[0, 0, 0], mod[0, 0, 1])
        for l, b in enumerate(self.blocks):
            q, k, v = qkv_norm_rope(
                b.qkv(hid),  # type: ignore
                b.q_norm,
                b.k_norm,
                cos,
                sin,
                c.heads,
                c.qk_eps,
            )
            hid, z = residual_adaln(
                b.o1(attention(q, k, v)),  # type: ignore
                z,
                mod[l, 0, 2],
                mod[l, 1, 0],
                mod[l, 1, 1],
            )
            q = head_norm(b.xq(hid), b.xq_norm, c.heads, c.qk_eps)  # type: ignore
            hid, z = residual_adaln(
                b.o2(ctx.attend(l, q)),  # type: ignore
                z,
                mod[l, 1, 2],
                mod[l, 2, 0],
                mod[l, 2, 1],
            )
            y = b.down(linear_gelu(hid, b.up.weight))  # type: ignore
            if l + 1 < c.layers:
                hid, z = residual_adaln(
                    y, z, mod[l, 2, 2], mod[l + 1, 0, 0], mod[l + 1, 0, 1]
                )
        shift, scale = fin.chunk(2, dim=-1)
        hid, _ = residual_adaln(y, z, mod[-1, 2, 2], shift, scale + 1)  # type: ignore
        return self.unpatchify(self.out_proj(hid), H, W)

    @torch.no_grad()
    def enable_fp8(self, fast_accum: bool = False) -> None:
        """Run inference with FP8 attention and MLP projections.

        Quantizes the current weights, so call it after merging any LoRA adapter and after the
        model is on its final device; call it again after changing the weights. Training and the
        reference forward keep using the original weights.

        Args:
            fast_accum: Accumulate with reduced precision for extra speed.

        Raises:
            RuntimeError: If the GPU does not support row-scaled FP8 matrix multiplication.
        """
        if not supported(self.patch.weight.device):
            raise RuntimeError("FP8 needs a Hopper or newer GPU")
        self.fp8 = [
            {n: quantize_weight(getattr(b, n).weight) for n in FP8_LAYERS}
            for b in self.blocks
        ]
        self.fp8_fast_accum = fast_accum

    def disable_fp8(self) -> None:
        """Go back to bf16 projections and free the quantized weights."""
        self.fp8 = None

    @torch.no_grad()
    def forward_fp8(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: Context,
        mods: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Inference forward pass with FP8 projections.

        Args:
            x: Noisy latent, shape `[B, H, W, C]`, on CUDA.
            t: Timesteps (sigmas), shape `[B]`; ignored if `mods` is given.
            ctx: Context from `encode_context`.
            mods: Optional precomputed `(mod, fin)` as returned by `modulation`, with batch size `B`.

        Returns:
            Predicted velocity, shape `[B, H, W, C]`.

        Raises:
            RuntimeError: If `enable_fp8` has not been called.
        """
        if self.fp8 is None:
            raise RuntimeError("call `enable_fp8()` first")
        c = self.cfg
        fast = self.fp8_fast_accum
        _, H, W, _ = x.shape  # B, H, W, C
        cos, sin = self.tables(H // 2, W // 2, x.device)
        z = self.patchify(x)
        mod, fin = mods if mods is not None else self.modulation(t, z.dtype)
        mod = mod.permute(1, 2, 3, 0, 4).contiguous()
        hq, hs = adaln_fp8(z, mod[0, 0, 0], mod[0, 0, 1])
        for l, b in enumerate(self.blocks):
            w = self.fp8[l]
            q, k, v = qkv_norm_rope(
                fp8_linear(hq, hs, w["qkv"], fast),
                b.q_norm,
                b.k_norm,
                cos,
                sin,
                c.heads,
                c.qk_eps,
            )
            oq, os_ = rowquant(attention(q, k, v))
            hq, hs, z = residual_adaln_fp8(
                fp8_linear(oq, os_, w["o1"], fast),
                z,
                mod[l, 0, 2],
                mod[l, 1, 0],
                mod[l, 1, 1],
            )
            q = head_norm(
                fp8_linear(hq, hs, w["xq"], fast), b.xq_norm, c.heads, c.qk_eps
            )  # type: ignore
            oq, os_ = rowquant(ctx.attend(l, q))
            hq, hs, z = residual_adaln_fp8(
                fp8_linear(oq, os_, w["o2"], fast),
                z,
                mod[l, 1, 2],
                mod[l, 2, 0],
                mod[l, 2, 1],
            )
            gq, gs = rowquant(fp8_linear(hq, hs, w["up"], fast), gelu=True)
            y = fp8_linear(gq, gs, w["down"], fast)
            if l + 1 < c.layers:
                hq, hs, z = residual_adaln_fp8(
                    y, z, mod[l, 2, 2], mod[l + 1, 0, 0], mod[l + 1, 0, 1]
                )
        shift, scale = fin.chunk(2, dim=-1)
        hid, _ = residual_adaln(y, z, mod[-1, 2, 2], shift, scale + 1)  # type: ignore
        return self.unpatchify(self.out_proj(hid), H, W)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: Context,
        mods: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Predict the velocity for a noisy latent.

        Dispatches to `forward_fp8` or `forward_fused` when `fused` is set, `x` is on CUDA, autograd is
        disabled and no LoRA adapter is attached; otherwise to `forward_reference`.

        Args:
            x: Noisy latent, shape `[B, H, W, C]`.
            t: Timesteps (sigmas), shape `[B]`; ignored if `mods` is given.
            ctx: Context from `encode_context`, with a batch size matching `x`.
            mods: Optional precomputed `(mod, fin)` as returned by `modulation`, with batch size `B`.

        Returns:
            Predicted velocity, shape `[B, H, W, C]`.
        """
        if (
            self.fused
            and x.is_cuda
            and not torch.is_grad_enabled()
            and not has_lora(self)
        ):
            if self.fp8 is not None:
                return self.forward_fp8(x, t, ctx, mods)
            return self.forward_fused(x, t, ctx, mods)
        return self.forward_reference(x, t, ctx, mods)

    @torch.no_grad()
    def denoise(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        x: torch.Tensor,
        steps: int,
        scale: float = 4.0,
        shift: float = 3.0,
        sampler: str = "euler",
        max_stage: int = 3,
        s_noise: float = 1.0,
        seeds: Sequence[int] | None = None,
        noise_scale: str = "er",
    ) -> torch.Tensor:
        """Sample with classifier-free guidance.

        Args:
            cond: Positive text embeddings, shape `[B, S, text_dim]`; its dtype is the model compute dtype.
            uncond: Negative/empty text embeddings, same shape and dtype as `cond`.
            x: Initial noise latent, float32, shape `[B, H, W, C]`.
            steps: Number of sampler steps; each costs one (guided) model evaluation.
            scale: Guidance scale (1 disables guidance).
            shift: Schedule shift passed to `sigmas`.
            sampler: `"euler"` (deterministic) or `"er_sde"` (stochastic).
            max_stage: Highest solver order for `"er_sde"` (1 to 3).
            s_noise: Multiplier on the noise `"er_sde"` injects; values below 1 shrink the result.
            seeds: One seed per batch element for the noise `"er_sde"` injects, so a sample does not
                depend on the rest of the batch; if None, the global torch RNG is used.
            noise_scale: For `"er_sde"`: `"er"` (stochastic) or `"ode"` (its noise-free limit).

        Returns:
            Denoised latent in float32, shape `[B, H, W, C]`.

        Raises:
            ValueError: If `sampler` is unknown or `max_stage` / `seeds` / `noise_scale` are invalid.
        """
        if sampler not in SAMPLERS:
            raise ValueError(f"unknown sampler {sampler!r}; expected one of {SAMPLERS}")
        dt = cond.dtype
        ctx = self.encode_context(torch.cat([cond, uncond]))
        sg = sigmas(steps, shift)
        sg_dev = sg.to(x.device)
        B = x.shape[0]
        # The schedule is known up front, so compute modulation for every step in one
        # batched call; the cond and uncond halves of a step share it.
        mod_all, fin_all = self.modulation(sg_dev[:steps], self.patch.weight.dtype)

        def velocity(i: int, z: torch.Tensor) -> torch.Tensor:
            mods = (
                mod_all[i : i + 1].expand(2 * B, -1, -1, -1, -1),
                fin_all[i : i + 1].expand(2 * B, -1),
            )
            v = self(
                torch.cat([z, z]).to(dt), sg_dev[i].expand(2 * B), ctx, mods
            ).float()
            vc, vu = v.chunk(2)
            return vu + scale * (vc - vu)

        if sampler == "euler":
            return euler(velocity, x, sg_dev)
        return er_sde(velocity, x, sg, max_stage, s_noise, seeds, noise_scale)


def group_dtype(
    key: str, dtype: torch.dtype, modulation_dtype: torch.dtype, norm_dtype: torch.dtype
) -> torch.dtype:
    """Pick the storage dtype of a checkpoint tensor by name.

    Args:
        key: Tensor name in the flat checkpoint.
        dtype: Dtype for ordinary weights.
        modulation_dtype: Dtype for the timestep/modulation weights (`MODULATION_KEYS`).
        norm_dtype: Dtype for per-block q/k norm weights.

    Returns:
        The dtype `key` should be loaded as.
    """
    if key.startswith(MODULATION_KEYS):
        return modulation_dtype
    if key.startswith("blocks.") and key.endswith("_norm"):
        return norm_dtype
    return dtype


def load(
    path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    modulation_dtype: torch.dtype = torch.float32,
    norm_dtype: torch.dtype = torch.float32,
    fused: bool = True,
    trim_context: bool = False,
    lora: str | None = None,
    lora_strength: float = 1.0,
) -> AnimaFlat:
    """Load an `anima-flat-1` safetensors checkpoint for inference.

    Args:
        path: Checkpoint file written by `convert_checkpoint`.
        device: Device to load the weights onto.
        dtype: Dtype of ordinary weights.
        modulation_dtype: Dtype of the modulation weights.
        norm_dtype: Dtype of the q/k norm weights.
        fused: Initial value of `AnimaFlat.fused`.
        trim_context: Initial value of `AnimaFlat.trim_context`.
        lora: Optional `anima-flat-lora-1` adapter file to merge into the weights.
        lora_strength: Multiplier applied to the adapter when merging.

    Returns:
        The model in eval mode with gradients disabled.

    Raises:
        ValueError: If the file's format metadata is not `anima-flat-1`, or `lora` is not a valid
            adapter file for this model.
    """
    with safe_open(path, framework="pt", device=device) as f:
        meta = f.metadata() or {}
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not an {FORMAT} checkpoint")
        cfg = FlatConfig.loads(meta["config"])
        sd = {
            k: f.get_tensor(k).to(group_dtype(k, dtype, modulation_dtype, norm_dtype))
            for k in f.keys()  # noqa: SIM118
        }
    with torch.device("meta"):
        model = AnimaFlat(cfg)
    model.load_state_dict(sd, strict=True, assign=True)
    model.requires_grad_(False)
    model.fused = fused
    model.trim_context = trim_context
    if lora is not None:
        merge_lora_file(model, lora, lora_strength)
    return model.eval()
