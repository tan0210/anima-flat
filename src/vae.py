# pyright: reportCallIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from .kernels.rms import add_rms_silu, rms_silu, shuffle_rms_silu

try:
    from ffpa_attn import ffpa_attn_func as _attend
except Exception:  # noqa: BLE001
    _attend = F.scaled_dot_product_attention

# Row `a` of A[phase] maps the three taps of a 3x3 kernel after a nearest 2x upsample onto the three
# taps of a 3x3 kernel over the original grid, for output rows/columns of parity `phase`.
_PHASE = torch.tensor(
    [[[1.0, 0, 0], [0, 1, 1], [0, 0, 0]], [[0, 0, 0], [1, 1, 0], [0, 0, 1]]]
)


PREFIX = "vae."


def _p(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t.detach().clone())


def _conv2d_weight(conv: nn.Module) -> torch.Tensor:
    w = conv.weight
    return (w[:, :, -1] if w.ndim == 5 else w).contiguous(
        memory_format=torch.channels_last
    )


def _put3d(conv: nn.Module, w: torch.Tensor, tap: int) -> None:
    with torch.no_grad():
        conv.weight[:, :, tap] = w.reshape(conv.weight[:, :, tap].shape).to(conv.weight)


def _put(dst: torch.Tensor, src: torch.Tensor) -> None:
    with torch.no_grad():
        dst.copy_(src.reshape(dst.shape).to(dst))


class _Res(nn.Module):
    def __init__(self, b: nn.Module) -> None:
        super().__init__()
        self.g1 = _p(b.norm1.gamma.reshape(-1))
        self.w1, self.b1 = _p(_conv2d_weight(b.conv1)), _p(b.conv1.bias)
        self.g2 = _p(b.norm2.gamma.reshape(-1))
        self.w2, self.b2 = _p(_conv2d_weight(b.conv2)), _p(b.conv2.bias)
        sc = b.conv_shortcut
        self.has_shortcut = not isinstance(sc, nn.Identity)
        if self.has_shortcut:
            self.sw, self.sb = _p(sc.weight[:, :, 0]), _p(sc.bias)

    def copy_to(self, b: nn.Module) -> None:
        _put(b.norm1.gamma, self.g1)
        _put3d(b.conv1, self.w1, 2)
        _put(b.conv1.bias, self.b1)
        _put(b.norm2.gamma, self.g2)
        _put3d(b.conv2, self.w2, 2)
        _put(b.conv2.bias, self.b2)
        if self.has_shortcut:
            _put3d(b.conv_shortcut, self.sw, 0)
            _put(b.conv_shortcut.bias, self.sb)

    def branch(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.conv2d(x, self.sw, self.sb) if self.has_shortcut else x
        c = F.conv2d(y, self.w1, self.b1, padding=1)
        z = rms_silu(c.permute(0, 2, 3, 1), self.g2).permute(0, 3, 1, 2)
        return F.conv2d(z, self.w2, self.b2, padding=1), h


class VAE(nn.Module):
    """Decoder of the Anima VAE (the Qwen-Image / Wan 2.1 architecture) for single images.

    A single frame makes every causal 3D convolution a 2D one, and the decoder runs on channels-last
    tensors with fused norm, activation, residual and upsampling steps. It is differentiable with
    respect to the latent and to all decoder weights. Attention uses `ffpa-attn` when it is
    installed, whose kernels cover the layer's head width of 384, and PyTorch's attention otherwise.

    Args:
        vae: A diffusers `AutoencoderKLQwenImage` or `AutoencoderKLWan` in the 2.1 layout, without
            patchify. Its weights are copied.

    Raises:
        ValueError: If the VAE uses the 2.2 residual layout or patchified input.
    """

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        dec = vae.decoder
        if getattr(vae.config, "patch_size", None) not in (None, 1) or getattr(
            vae.config, "is_residual", False
        ):
            raise ValueError("only the Wan 2.1 VAE layout is supported")
        self.register_buffer(
            "mean",
            torch.tensor(vae.config.latents_mean, dtype=torch.float32),  # type: ignore
        )
        self.register_buffer(
            "std",
            torch.tensor(vae.config.latents_std, dtype=torch.float32),  # type: ignore
        )
        self.post_w, self.post_b = (
            _p(vae.post_quant_conv.weight[:, :, 0]),  # type: ignore
            _p(vae.post_quant_conv.bias),  # type: ignore
        )
        self.in_w, self.in_b = _p(_conv2d_weight(dec.conv_in)), _p(dec.conv_in.bias)  # type: ignore
        mid = dec.mid_block  # type: ignore
        self.mid0, self.mid1 = _Res(mid.resnets[0]), _Res(mid.resnets[1])  # type: ignore
        attn = mid.attentions[0]  # type: ignore
        self.attn_g = _p(attn.norm.gamma.reshape(-1))  # type: ignore
        self.qkv_w, self.qkv_b = _p(attn.to_qkv.weight), _p(attn.to_qkv.bias)  # type: ignore
        self.proj_w, self.proj_b = _p(attn.proj.weight), _p(attn.proj.bias)  # type: ignore
        self.blocks = nn.ModuleList()
        self.up_w = nn.ParameterList()
        self.up_b = nn.ParameterList()
        for ub in dec.up_blocks:  # type: ignore
            self.blocks.append(nn.ModuleList(_Res(r) for r in ub.resnets))
            if ub.upsamplers is not None:
                conv = ub.upsamplers[0].resample[1]
                self.up_w.append(_p(conv.weight))
                self.up_b.append(_p(conv.bias))
        self.out_g = _p(dec.norm_out.gamma.reshape(-1))  # type: ignore
        self.out_w, self.out_b = (
            _p(_conv2d_weight(dec.conv_out)),  # type: ignore
            _p(dec.conv_out.bias),  # type: ignore
        )

    @classmethod
    def from_state_dict(cls, sd: dict[str, torch.Tensor]) -> "VAE":
        """Rebuild the decoder from the tensors of `state_dict()`.

        Args:
            sd: Tensors as returned by `state_dict()`, including `mean` and `std`.

        Returns:
            The decoder holding copies of `sd`'s tensors.
        """
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.register_buffer("mean", sd["mean"].clone())
        m.register_buffer("std", sd["std"].clone())

        def make(prefix: str) -> _Res:
            r = _Res.__new__(_Res)
            nn.Module.__init__(r)
            for n in ("g1", "w1", "b1", "g2", "w2", "b2"):
                setattr(r, n, _p(sd[prefix + n]))
            r.has_shortcut = prefix + "sw" in sd
            if r.has_shortcut:
                r.sw, r.sb = _p(sd[prefix + "sw"]), _p(sd[prefix + "sb"])
            return r

        for n in ("post_w", "post_b", "in_w", "in_b", "attn_g", "qkv_w", "qkv_b"):
            setattr(m, n, _p(sd[n]))
        for n in ("proj_w", "proj_b", "out_g", "out_w", "out_b"):
            setattr(m, n, _p(sd[n]))
        m.mid0, m.mid1 = make("mid0."), make("mid1.")
        m.blocks = nn.ModuleList()
        i = 0
        while f"blocks.{i}.0.g1" in sd:
            j = 0
            res = []
            while f"blocks.{i}.{j}.g1" in sd:
                res.append(make(f"blocks.{i}.{j}."))
                j += 1
            m.blocks.append(nn.ModuleList(res))
            i += 1
        m.up_w, m.up_b = nn.ParameterList(), nn.ParameterList()
        k = 0
        while f"up_w.{k}" in sd:
            m.up_w.append(_p(sd[f"up_w.{k}"]))
            m.up_b.append(_p(sd[f"up_b.{k}"]))
            k += 1
        return m

    def copy_to(self, vae: nn.Module) -> None:
        """Write the decoder weights into a diffusers VAE of the same architecture.

        Only the temporal tap that single frames use is overwritten in each 3D convolution, so the
        result is a valid diffusers VAE that decodes images like this module.

        Args:
            vae: The target `AutoencoderKLQwenImage` or `AutoencoderKLWan`, modified in place.
        """
        dec = vae.decoder
        _put3d(vae.post_quant_conv, self.post_w, 0)  # type: ignore
        _put(vae.post_quant_conv.bias, self.post_b)  # type: ignore
        _put3d(dec.conv_in, self.in_w, 2)  # type: ignore
        _put(dec.conv_in.bias, self.in_b)  # type: ignore
        mid = dec.mid_block  # type: ignore
        self.mid0.copy_to(mid.resnets[0])  # type: ignore
        self.mid1.copy_to(mid.resnets[1])  # type: ignore
        attn = mid.attentions[0]  # type: ignore
        _put(attn.norm.gamma, self.attn_g)  # type: ignore
        _put(attn.to_qkv.weight, self.qkv_w)  # type: ignore
        _put(attn.to_qkv.bias, self.qkv_b)  # type: ignore
        _put(attn.proj.weight, self.proj_w)  # type: ignore
        _put(attn.proj.bias, self.proj_b)  # type: ignore
        i = 0
        for ub, block in zip(dec.up_blocks, self.blocks):  # type: ignore
            for target, res in zip(ub.resnets, block):
                res.copy_to(target)
            if ub.upsamplers is not None:
                conv = ub.upsamplers[0].resample[1]
                _put(conv.weight, self.up_w[i])
                _put(conv.bias, self.up_b[i])
                i += 1
        _put(dec.norm_out.gamma, self.out_g)  # type: ignore
        _put3d(dec.conv_out, self.out_w, 2)  # type: ignore
        _put(dec.conv_out.bias, self.out_b)  # type: ignore

    @staticmethod
    def _subpixel(
        w: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 3x3 conv after a nearest 2x upsample == one 3x3 conv with 4x the outputs, then depth-to-space.
        A = _PHASE.to(w)
        w4 = torch.einsum("aij,bkl,ocjl->abocik", A, A, w)
        co, ci = w.shape[:2]
        w4 = w4.reshape(4 * co, ci, 3, 3).contiguous(memory_format=torch.channels_last)
        return w4, b.repeat(4)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        n = (
            F.normalize(x.float(), dim=1).to(x.dtype)
            * C**0.5
            * self.attn_g.view(1, C, 1, 1)
        )
        qkv = F.conv2d(n, self.qkv_w, self.qkv_b).reshape(B, 1, 3 * C, H * W)
        q, k, v = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        o = _attend(q.contiguous(), k.contiguous(), v.contiguous())
        o = o.squeeze(1).permute(0, 2, 1).reshape(B, C, H, W)
        return F.conv2d(o, self.proj_w, self.proj_b) + x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode denormalized latents to images.

        Args:
            z: Denormalized latents, shape `[B, z_dim, H, W]`, CUDA.

        Returns:
            Images of shape `[B, 3, 8 * H, 8 * W]`, nominally in `[-1, 1]` (not clamped).
        """
        nhwc = lambda t: t.permute(0, 2, 3, 1)
        nchw = lambda t: t.permute(0, 3, 1, 2)
        z = z.contiguous(memory_format=torch.channels_last)
        x = F.conv2d(
            F.conv2d(z, self.post_w, self.post_b), self.in_w, self.in_b, padding=1
        )
        y = nchw(rms_silu(nhwc(x), self.mid0.g1))
        c, h = self.mid0.branch(x, y)
        x = self._attention(c + h)
        y = nchw(rms_silu(nhwc(x), self.mid1.g1))
        c, h = self.mid1.branch(x, y)
        stages = [(res, i) for i, block in enumerate(self.blocks) for res in block]  # type: ignore
        upsample = 0
        for k, (res, i) in enumerate(stages):
            if k == 0:
                s, y = add_rms_silu(nhwc(c), nhwc(h), res.g1)
                x, y = nchw(s), nchw(y)
            c, h = res.branch(x, y)
            if k + 1 == len(stages):
                s, y = add_rms_silu(nhwc(c), nhwc(h), self.out_g)
            elif stages[k + 1][1] == i:
                s, y = add_rms_silu(nhwc(c), nhwc(h), stages[k + 1][0].g1)
            else:
                w4, b4 = self._subpixel(self.up_w[upsample], self.up_b[upsample])
                upsample += 1
                u = F.conv2d(c + h, w4, b4, padding=1)
                s, y = shuffle_rms_silu(nhwc(u), stages[k + 1][0].g1)
            x, y = nchw(s), nchw(y)
        return F.conv2d(y, self.out_w, self.out_b, padding=1)

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode sampler latents to images.

        Args:
            x: Latents in the sampler's normalized space, shape `[B, H, W, C]`, CUDA.

        Returns:
            Images of shape `[B, 3, 8 * H, 8 * W]`, clamped to `[-1, 1]`, in the weights' dtype.
        """
        z = x.permute(0, 3, 1, 2).float() * self.std.view(1, -1, 1, 1) + self.mean.view(  # type: ignore
            1, -1, 1, 1
        )
        return self(z.to(self.in_w.dtype)).clamp(-1, 1)


def load_vae(
    path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> VAE | None:
    """Load the decoder stored in an `anima-flat-1` checkpoint.

    Args:
        path: Checkpoint file written by `convert_checkpoint`.
        device: Device to load the weights onto.
        dtype: Dtype of the decoder weights.

    Returns:
        The decoder in eval mode, or `None` if the file holds no decoder weights.
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith(PREFIX)]  # noqa: SIM118
        if not keys:
            return None
        sd = {k[len(PREFIX) :]: f.get_tensor(k) for k in keys}
    vae = VAE.from_state_dict(sd)
    vae.requires_grad_(False)
    return vae.to(device, dtype).eval()
