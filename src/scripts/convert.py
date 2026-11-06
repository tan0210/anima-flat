import glob
import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

from ..config import FORMAT, FlatConfig


def config_from_diffusers(
    sd: dict[str, torch.Tensor], hf: dict, qk_eps: float = 1e-5
) -> FlatConfig:
    """Infer a `FlatConfig` from a diffusers state dict and its `config.json`.

    Args:
        sd: Diffusers-layout state dict.
        hf: Parsed diffusers `config.json`.
        qk_eps: Epsilon of the q/k RMS norms.

    Returns:
        The inferred config.

    Raises:
        ValueError: If the checkpoint uses a learned positional embedding, a patch size other
            than `[1, 2, 2]`, or no padding-mask channel.
    """
    if hf.get("extra_pos_embed_type") is not None:
        raise ValueError(
            "checkpoints with a learned positional embedding are not supported"
        )
    if list(hf.get("patch_size", [1, 2, 2])) != [1, 2, 2] or not hf.get(
        "concat_padding_mask", True
    ):
        raise ValueError("expected patch_size [1, 2, 2] and concat_padding_mask")
    rope = hf.get("rope_scale", [1.0, 1.0, 1.0])
    D = sd["patch_embed.proj.weight"].shape[0]
    hd = sd["transformer_blocks.0.attn1.norm_q.weight"].shape[0]
    layers = 1 + max(
        int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks.")
    )
    return FlatConfig(
        layers=layers,
        heads=D // hd,
        head_dim=hd,
        text_dim=sd["transformer_blocks.0.attn2.to_k.weight"].shape[1],
        lora_dim=sd["transformer_blocks.0.norm1.linear_1.weight"].shape[0],
        mlp_dim=sd["transformer_blocks.0.ff.net.0.proj.weight"].shape[0],
        in_ch=sd["proj_out.weight"].shape[0] // 4,
        qk_eps=qk_eps,
        rope_scale=(float(rope[1]), float(rope[2])),
    )


def from_diffusers(
    sd: dict[str, torch.Tensor], cfg: FlatConfig
) -> dict[str, torch.Tensor]:
    """Convert a diffusers state dict to the flat layout.

    Fuses q/k/v (and cross-attention k/v), stacks the adaLN modulation weights, and drops
    the unused padding-mask channel of the patch embedding.

    Args:
        sd: Diffusers-layout state dict.
        cfg: Config describing `sd` (see `config_from_diffusers`).

    Returns:
        Flat-layout state dict of contiguous tensors.
    """
    L, C = cfg.layers, cfg.in_ch
    pw = sd["patch_embed.proj.weight"]
    D = pw.shape[0]
    names = [(l, j) for l in range(L) for j in (1, 2, 3)]
    flat = {
        "patch.weight": pw.view(D, C + 1, 2, 2)[:, :C]
        .permute(0, 2, 3, 1)
        .reshape(D, 4 * C),
        "t_lin1.weight": sd["time_embed.t_embedder.linear_1.weight"],
        "t_lin2.weight": sd["time_embed.t_embedder.linear_2.weight"],
        "t_norm": sd["time_embed.norm.weight"],
        "mod_w1": torch.cat(
            [sd[f"transformer_blocks.{l}.norm{j}.linear_1.weight"] for l, j in names]
            + [sd["norm_out.linear_1.weight"]]
        ),
        "mod_w2": torch.stack(
            [sd[f"transformer_blocks.{l}.norm{j}.linear_2.weight"] for l, j in names]
        ),
        "out_w2": sd["norm_out.linear_2.weight"],
        "out_proj.weight": sd["proj_out.weight"],
    }
    for l in range(L):
        p, b = f"transformer_blocks.{l}.", f"blocks.{l}."
        flat[b + "qkv.weight"] = torch.cat(
            [sd[p + f"attn1.to_{n}.weight"] for n in "qkv"]
        )
        flat[b + "q_norm"] = sd[p + "attn1.norm_q.weight"]
        flat[b + "k_norm"] = sd[p + "attn1.norm_k.weight"]
        flat[b + "o1.weight"] = sd[p + "attn1.to_out.0.weight"]
        flat[b + "xq.weight"] = sd[p + "attn2.to_q.weight"]
        flat[b + "xkv.weight"] = torch.cat(
            [sd[p + "attn2.to_k.weight"], sd[p + "attn2.to_v.weight"]]
        )
        flat[b + "xq_norm"] = sd[p + "attn2.norm_q.weight"]
        flat[b + "xk_norm"] = sd[p + "attn2.norm_k.weight"]
        flat[b + "o2.weight"] = sd[p + "attn2.to_out.0.weight"]
        flat[b + "up.weight"] = sd[p + "ff.net.0.proj.weight"]
        flat[b + "down.weight"] = sd[p + "ff.net.2.weight"]
    return {k: v.contiguous() for k, v in flat.items()}


def to_diffusers(
    flat: dict[str, torch.Tensor], cfg: FlatConfig
) -> dict[str, torch.Tensor]:
    """Convert a flat state dict back to the diffusers layout.

    Inverse of `from_diffusers`; the padding-mask channel of the patch embedding is
    restored as zeros.

    Args:
        flat: Flat-layout state dict.
        cfg: Config describing `flat`.

    Returns:
        Diffusers-layout state dict of contiguous tensors.
    """
    L, C, R, D = cfg.layers, cfg.in_ch, cfg.lora_dim, cfg.dim
    w = flat["patch.weight"].view(D, 2, 2, C).permute(0, 3, 1, 2)
    w = torch.cat([w, w.new_zeros(D, 1, 2, 2)], dim=1).reshape(D, (C + 1) * 4)
    w1 = flat["mod_w1"].view(3 * L + 1, R, D)
    sd = {
        "patch_embed.proj.weight": w,
        "time_embed.t_embedder.linear_1.weight": flat["t_lin1.weight"],
        "time_embed.t_embedder.linear_2.weight": flat["t_lin2.weight"],
        "time_embed.norm.weight": flat["t_norm"],
        "norm_out.linear_1.weight": w1[-1],
        "norm_out.linear_2.weight": flat["out_w2"],
        "proj_out.weight": flat["out_proj.weight"],
    }
    for l in range(L):
        p, b = f"transformer_blocks.{l}.", f"blocks.{l}."
        for n, t in zip("qkv", flat[b + "qkv.weight"].chunk(3)):
            sd[p + f"attn1.to_{n}.weight"] = t
        xk, xv = flat[b + "xkv.weight"].chunk(2)
        sd[p + "attn1.norm_q.weight"] = flat[b + "q_norm"]
        sd[p + "attn1.norm_k.weight"] = flat[b + "k_norm"]
        sd[p + "attn1.to_out.0.weight"] = flat[b + "o1.weight"]
        sd[p + "attn2.to_q.weight"] = flat[b + "xq.weight"]
        sd[p + "attn2.to_k.weight"] = xk
        sd[p + "attn2.to_v.weight"] = xv
        sd[p + "attn2.norm_q.weight"] = flat[b + "xq_norm"]
        sd[p + "attn2.norm_k.weight"] = flat[b + "xk_norm"]
        sd[p + "attn2.to_out.0.weight"] = flat[b + "o2.weight"]
        sd[p + "ff.net.0.proj.weight"] = flat[b + "up.weight"]
        sd[p + "ff.net.2.weight"] = flat[b + "down.weight"]
        for j in range(3):
            sd[p + f"norm{j + 1}.linear_1.weight"] = w1[3 * l + j]
            sd[p + f"norm{j + 1}.linear_2.weight"] = flat["mod_w2"][3 * l + j]
    return {k: v.contiguous() for k, v in sd.items()}


def convert_checkpoint(
    src_dir: str | os.PathLike,
    dst_path: str | os.PathLike,
    qk_eps: float = 1e-5,
    dtype: torch.dtype | None = None,
) -> FlatConfig:
    """Convert a diffusers transformer directory into an `anima-flat-1` safetensors file.

    Args:
        src_dir: Directory containing `config.json` and one or more `*.safetensors` shards.
        dst_path: Output safetensors path.
        qk_eps: Epsilon of the q/k RMS norms, stored in the config.
        dtype: If given, cast all tensors to this dtype before saving.

    Returns:
        The config written to the file's metadata.

    Raises:
        ValueError: If the checkpoint is unsupported (see `config_from_diffusers`).
    """
    with open(os.path.join(src_dir, "config.json")) as f:
        hf = json.load(f)
    sd = {}
    for path in sorted(glob.glob(os.path.join(src_dir, "*.safetensors"))):
        sd.update(load_file(path))
    cfg = config_from_diffusers(sd, hf, qk_eps)
    flat = from_diffusers(sd, cfg)
    if dtype is not None:
        flat = {k: v.to(dtype).contiguous() for k, v in flat.items()}
    save_file(flat, dst_path, metadata={"format": FORMAT, "config": cfg.dumps()})
    return cfg

PROJ = {
    "self_attn.q_proj": ("qkv", 0),
    "self_attn.k_proj": ("qkv", 1),
    "self_attn.v_proj": ("qkv", 2),
    "self_attn.output_proj": ("o1", 0),
    "cross_attn.q_proj": ("xq", 0),
    "cross_attn.k_proj": ("xkv", 0),
    "cross_attn.v_proj": ("xkv", 1),
    "cross_attn.output_proj": ("o2", 0),
    "mlp.layer1": ("up", 0),
    "mlp.layer2": ("down", 0),
}
DIFFUSERS = {
    "attn1.to_q": "self_attn.q_proj",
    "attn1.to_k": "self_attn.k_proj",
    "attn1.to_v": "self_attn.v_proj",
    "attn1.to_out.0": "self_attn.output_proj",
    "attn2.to_q": "cross_attn.q_proj",
    "attn2.to_k": "cross_attn.k_proj",
    "attn2.to_v": "cross_attn.v_proj",
    "attn2.to_out.0": "cross_attn.output_proj",
    "ff.net.0.proj": "mlp.layer1",
    "ff.net.2": "mlp.layer2",
}
KOHYA = {k.replace(".", "_"): k for k in PROJ}
PARTS = {"qkv": 3, "xkv": 2, "xq": 1, "o1": 1, "o2": 1, "up": 1, "down": 1}
PREFIX = r"^(?:model\.diffusion_model\.|diffusion_model\.|base_model\.model\.transformer\.|base_model\.model\.|transformer\.)?"

def split_key(k: str) -> tuple[str | None, str | None]:
    """Split a LoRA tensor name into module stem and role.

    Args:
        k: Tensor name (PEFT `lora_A/lora_B`, kohya `lora_down/lora_up`, or `.alpha`).

    Returns:
        `(stem, kind)` with `kind` in `{"A", "B", "alpha"}`, or `(None, None)` if `k` is not a LoRA tensor.
    """
    m = re.search(r"\.lora_(A|B)(?:\.[^.]+)?\.weight$", k)
    if m:
        return k[:m.start()], m.group(1)
    m = re.search(r"\.lora_(down|up)\.weight$", k)
    if m:
        return k[:m.start()], "A" if m.group(1) == "down" else "B"
    if k.endswith(".alpha"):
        return k[:-6], "alpha"
    return None, None


def locate(stem: str) -> tuple[int, str] | None:
    """Map a LoRA module stem to a block index and canonical projection name.

    Accepts dotted (native/diffusers/PEFT) and underscore (kohya `lora_unet_...`) stems.

    Args:
        stem: Module stem from `split_key`.

    Returns:
        `(layer, name)` with `name` a key of `PROJ`, or None if the module is not supported.
    """
    m = re.match(PREFIX + r"(?:transformer_)?blocks\.(\d+)\.(.+)$", stem)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        rest = DIFFUSERS.get(rest, rest)
        return (layer, rest) if rest in PROJ else None
    m = re.match(r"^lora_unet_(?:transformer_)?blocks_(\d+)_(.+)$", stem)
    if m and m.group(2) in KOHYA:
        return int(m.group(1)), KOHYA[m.group(2)]
    return None


def blockdiag(bs: list[torch.Tensor]) -> torch.Tensor:
    """Assemble 2D matrices into a block-diagonal matrix.

    Args:
        bs: Non-empty list of matrices (zero-sized dimensions allowed), all on one device and dtype.

    Returns:
        Matrix of shape `[sum of rows, sum of cols]` with each input on the diagonal and zeros elsewhere.
    """
    rows = sum(b.shape[0] for b in bs)
    cols = sum(b.shape[1] for b in bs)
    out = bs[0].new_zeros(rows, cols)
    r = c = 0
    for b in bs:
        out[r:r + b.shape[0], c:c + b.shape[1]] = b
        r += b.shape[0]
        c += b.shape[1]
    return out


def convert_lora(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Convert a LoRA state dict to the flat layout.

    Per-projection factors are fused to match the flat weights: q/k/v (and cross k/v) become one
    `lora_A` (stacked along rank) and a block-diagonal `lora_B`. `alpha / rank` is folded into
    `lora_B` (alpha defaults to the rank). Missing projections of a fused group get zero-rank blocks.

    Args:
        sd: LoRA state dict (PEFT, kohya, or native naming).

    Returns:
        `(out, skipped)`: float32 CPU tensors `blocks.{layer}.{target}.lora_A` / `.lora_B`, and the
        names of unrecognized keys that were ignored.

    Raises:
        ValueError: If a module has an A without a B (or vice versa).
    """
    found, skipped = {}, []
    for k, v in sd.items():
        stem, kind = split_key(k)
        loc = locate(stem) if stem is not None else None
        if loc is None:
            skipped.append(k)
            continue
        found.setdefault(loc, {})[kind] = v.detach().float().cpu()
    groups = {}
    for (layer, name), t in found.items():
        target, idx = PROJ[name]
        groups.setdefault((layer, target), {})[idx] = t
    out = {}
    for (layer, target), parts in sorted(groups.items()):
        for p in parts.values():
            if "A" not in p or "B" not in p:
                raise ValueError(f"block {layer} {target}: incomplete A/B pair")
        ref = next(iter(parts.values()))
        d_in, d_out = ref["A"].shape[1], ref["B"].shape[0]
        As, Bs = [], []
        for i in range(PARTS[target]):
            if i in parts:
                a, b = parts[i]["A"], parts[i]["B"]
                r = a.shape[0]
                alpha = parts[i]["alpha"].item() if "alpha" in parts[i] else float(r)
                As.append(a)
                Bs.append(b * (alpha / r))
            else:
                As.append(torch.zeros(0, d_in))
                Bs.append(torch.zeros(d_out, 0))
        out[f"blocks.{layer}.{target}.lora_A"] = torch.cat(As, 0).contiguous()
        out[f"blocks.{layer}.{target}.lora_B"] = blockdiag(Bs).contiguous()
    print(f"converted {len(out) // 2} modules, skipped {len(skipped)} keys")
    for k in skipped[:8]:
        print("  skipped", k)
    return out, skipped


def convert_lora_file(src: str | os.PathLike, dst: str | os.PathLike) -> list[str]:
    """Convert a LoRA safetensors file to the `anima-flat-lora-1` format.

    Args:
        src: Input LoRA safetensors path.
        dst: Output safetensors path.

    Returns:
        Names of the keys in `src` that were skipped.

    Raises:
        ValueError: If a module has an incomplete A/B pair (see `convert_lora`).
    """
    out, skipped = convert_lora(load_file(src))
    save_file(out, dst, metadata={"format": "anima-flat-lora-1"})
    return skipped


@torch.no_grad()
def merge_lora(flat: dict[str, torch.Tensor], lora: dict[str, torch.Tensor],
               strength: float = 1.0) -> dict[str, torch.Tensor]:
    """Merge a flat-layout LoRA into flat model weights.

    Computes `W + strength * (B @ A)` in float32 and casts back to each weight's dtype.

    Args:
        flat: Flat-layout model state dict; not modified.
        lora: Flat-layout LoRA from `convert_lora`.
        strength: LoRA multiplier.

    Returns:
        A new state dict with the LoRA-targeted weights replaced by merged ones.

    Raises:
        KeyError: If a LoRA module has no matching weight in `flat`.
    """
    out = dict(flat)
    for k in lora: # noqa: PLC0206
        if not k.endswith(".lora_A"):
            continue
        base = k[:-7]
        w = out[base + ".weight"]
        a = lora[k].to(w.device)
        b = lora[base + ".lora_B"].to(w.device)
        out[base + ".weight"] = (w.float() + strength * (b @ a)).to(w.dtype)
    return out
