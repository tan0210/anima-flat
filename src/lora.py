import math
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

LORA_FORMAT = "anima-flat-lora-1"
PARTS = {"qkv": 3, "xkv": 2, "xq": 1, "o1": 1, "o2": 1, "up": 1, "down": 1}
TARGETS = tuple(PARTS)


@torch.library.custom_op("anima_flat::lora_add", mutates_args=())
def _lora_add(
    base: torch.Tensor,
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    a, b = A.to(x.dtype), B.to(x.dtype)
    parts, o, r = b.shape
    low = x @ a.T
    if parts == 1:
        return torch.addmm(base, low, b[0].T, alpha=scale)
    out = base.clone()
    for i in range(parts):
        out[:, i * o : (i + 1) * o].addmm_(
            low[:, i * r : (i + 1) * r], b[i].T, alpha=scale
        )
    return out


@_lora_add.register_fake
def _(base, x, A, B, scale):
    return torch.empty_like(base)


def _setup_lora_add(ctx, inputs, output):
    _, x, A, B, scale = inputs
    ctx.save_for_backward(x, A, B)
    ctx.scale = scale


def _bwd_lora_add(ctx, g):
    x, A, B = ctx.saved_tensors
    a, b = A.to(x.dtype), B.to(x.dtype)
    parts, o, r = b.shape
    low = x @ a.T
    dlow = torch.empty_like(low)
    dB = []
    for i in range(parts):
        gi = g[:, i * o : (i + 1) * o]
        dlow[:, i * r : (i + 1) * r] = gi @ b[i]
        if ctx.needs_input_grad[3]:
            dB.append(gi.T @ low[:, i * r : (i + 1) * r])
    dlow.mul_(ctx.scale)
    dx = dlow @ a if ctx.needs_input_grad[1] else None
    dA = (dlow.T @ x).float() if ctx.needs_input_grad[2] else None
    dBs = (torch.stack(dB) * ctx.scale).float() if dB else None
    return g, dx, dA, dBs, None


_lora_add.register_autograd(_bwd_lora_add, setup_context=_setup_lora_add)


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update.

    For fused projections (`parts` > 1, e.g. `qkv`) each part gets its own independent rank-`rank`
    update, exactly as if q, k and v were adapted separately.

    Args:
        base: The layer to adapt; its weight is not modified.
        parts: Number of equal-sized output slices adapted independently.
        rank: Rank of each update.
        alpha: Scaling numerator; the update is scaled by `alpha / rank`.
        dropout: Dropout probability on the input of the low-rank path.

    Raises:
        ValueError: If `rank < 1` or the output width is not divisible by `parts`.
    """

    def __init__(
        self,
        base: nn.Linear,
        parts: int = 1,
        rank: int = 16,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if base.out_features % parts:
            raise ValueError(
                f"{base.out_features} outputs not divisible into {parts} parts"
            )
        self.base, self.parts, self.rank = base, parts, rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scale = self.alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev = base.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(parts * rank, base.in_features, device=dev, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                parts, base.out_features // parts, rank, device=dev, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the adapted layer.

        Args:
            x: Input, shape `[..., in_features]`.

        Returns:
            Output in the dtype of `x`, shape `[..., out_features]`.
        """
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        out = self.base(flat)
        out = _lora_add(out, self.drop(flat), self.lora_A, self.lora_B, self.scale)
        return out.view(*lead, out.shape[-1])

    def delta_weight(self) -> torch.Tensor:
        """Weight update this adapter adds to the base weight.

        Returns:
            Float32 tensor of shape `[out_features, in_features]`, including the `alpha / rank` scale.
        """
        b = torch.block_diag(*self.lora_B.unbind(0))
        return (b.float() @ self.lora_A.float()) * self.scale


def has_lora(model: nn.Module) -> bool:
    """Report whether any layer of `model` carries an adapter.

    Args:
        model: The model to inspect.

    Returns:
        True if at least one `LoRALinear` is present.
    """
    return any(isinstance(m, LoRALinear) for m in model.modules())


def add_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float | None = None,
    targets: tuple[str, ...] = TARGETS,
    dropout: float = 0.0,
    layers: range | list[int] | None = None,
) -> list[nn.Parameter]:
    """Attach trainable adapters to the model and freeze everything else.

    Args:
        model: An `AnimaFlat` model.
        rank: Rank of each update.
        alpha: Scaling numerator; defaults to `rank`.
        targets: Projections to adapt, any of `TARGETS`.
        dropout: Dropout probability on the adapter input.
        layers: Block indices to adapt; defaults to all blocks.

    Returns:
        The trainable adapter parameters, in a stable order.

    Raises:
        ValueError: If a target is unknown or a selected layer already has an adapter.
    """
    unknown = [t for t in targets if t not in PARTS]
    if unknown:
        raise ValueError(f"unknown targets {unknown}; expected a subset of {TARGETS}")
    model.requires_grad_(False)
    idx = range(len(model.blocks)) if layers is None else layers  # type: ignore
    for i in idx:
        for name in targets:
            _wrap(model.blocks[i], name, rank, alpha, dropout)  # type: ignore
    return lora_parameters(model)


def _wrap(
    block: nn.Module, name: str, rank: int, alpha: float | None, dropout: float
) -> LoRALinear:
    base = getattr(block, name)
    if isinstance(base, LoRALinear):
        raise ValueError(f"{name} already has an adapter")  # noqa: TRY004
    mod = LoRALinear(base, PARTS[name], rank, alpha, dropout)
    setattr(block, name, mod)
    return mod


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect the trainable adapter parameters.

    Args:
        model: The model.

    Returns:
        All adapter parameters, in module order.
    """
    return [
        p
        for m in model.modules()
        if isinstance(m, LoRALinear)
        for p in (m.lora_A, m.lora_B)
    ]


@torch.no_grad()
def remove_lora(model: nn.Module, merge: bool = True) -> None:
    """Detach all adapters from the model.

    Args:
        model: The model.
        merge: If True, add each adapter's update into the base weight (cast to its dtype) first.
    """
    for block in model.blocks:  # type: ignore
        for name in TARGETS:
            m = getattr(block, name)
            if not isinstance(m, LoRALinear):
                continue
            if merge:
                w = m.base.weight
                w.copy_((w.float() + m.delta_weight().to(w.device)).to(w.dtype))
            setattr(block, name, m.base)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Export the adapters in the `anima-flat-lora-1` layout.

    Args:
        model: The model.

    Returns:
        Float32 CPU tensors `blocks.{layer}.{target}.lora_A` and `.lora_B`, with `alpha / rank`
        already applied to `lora_B`.
    """
    out = {}
    for i, block in enumerate(model.blocks):  # type: ignore
        for name in TARGETS:
            m = getattr(block, name)
            if isinstance(m, LoRALinear):
                b = torch.block_diag(*m.lora_B.detach().unbind(0)).float() * m.scale
                out[f"blocks.{i}.{name}.lora_A"] = (
                    m.lora_A.detach().float().cpu().contiguous()
                )
                out[f"blocks.{i}.{name}.lora_B"] = b.cpu().contiguous()
    return out


def save_lora(model: nn.Module, path: str | os.PathLike) -> None:
    """Save the adapters to a safetensors file.

    The file can be loaded back with `load_lora` or merged into a model with
    `merge_lora_file`.

    Args:
        model: The model.
        path: Output path.

    Raises:
        ValueError: If the model has no adapters.
    """
    sd = lora_state_dict(model)
    if not sd:
        raise ValueError("model has no adapters")
    first = next(m for m in model.modules() if isinstance(m, LoRALinear))
    meta = {"format": LORA_FORMAT, "rank": str(first.rank), "alpha": str(first.alpha)}
    save_file(sd, path, metadata=meta)


def load_lora(
    model: nn.Module, path: str | os.PathLike, dropout: float = 0.0
) -> list[nn.Parameter]:
    """Attach adapters from a file written by `save_lora` and freeze everything else.

    Args:
        model: An `AnimaFlat` model without adapters on the layers in the file.
        path: Adapter file.
        dropout: Dropout probability on the adapter input.

    Returns:
        The trainable adapter parameters.

    Raises:
        ValueError: If the file is not an `anima-flat-lora-1` file, its layer names do not match
            the model, or its per-projection ranks are unequal.
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("format") != LORA_FORMAT:
            raise ValueError(f"{path} is not an {LORA_FORMAT} file")
        sd = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    model.requires_grad_(False)
    for key in sorted(k for k in sd if k.endswith(".lora_A")):
        _, layer, name, _ = key.split(".")
        i = int(layer)
        if name not in PARTS or i >= len(model.blocks):  # type: ignore
            raise ValueError(f"{key} does not match the model")
        parts = PARTS[name]
        a, b = sd[key], sd[key[:-1] + "B"]
        if a.shape[0] % parts:
            raise ValueError(f"{key}: unequal per-projection ranks are not supported")
        rank = a.shape[0] // parts
        alpha = (
            float(meta.get("alpha", rank))
            if meta.get("rank") == str(rank)
            else float(rank)
        )
        mod = _wrap(model.blocks[i], name, rank, alpha, dropout)  # type: ignore
        out = b.shape[0] // parts
        blocks = [
            b[j * out : (j + 1) * out, j * rank : (j + 1) * rank] for j in range(parts)
        ]
        with torch.no_grad():
            mod.lora_A.copy_(a)
            mod.lora_B.copy_(torch.stack(blocks) / mod.scale)
    return lora_parameters(model)


@torch.no_grad()
def merge_lora_file(
    model: nn.Module, path: str | os.PathLike, strength: float = 1.0
) -> None:
    """Merge an adapter file directly into the model's weights.

    Accepts any `anima-flat-lora-1` file, including ones converted from other formats.

    Args:
        model: An `AnimaFlat` model.
        path: Adapter file.
        strength: Multiplier applied to the adapter.

    Raises:
        ValueError: If the file is not an `anima-flat-lora-1` file or its layer names do not
            match the model.
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("format") != LORA_FORMAT:
            raise ValueError(f"{path} is not an {LORA_FORMAT} file")
        for key in f.keys():  # noqa: SIM118
            if not key.endswith(".lora_A"):
                continue
            _, layer, name, _ = key.split(".")
            if name not in PARTS or int(layer) >= len(model.blocks):  # type: ignore
                raise ValueError(f"{key} does not match the model")
            layer_mod = getattr(model.blocks[int(layer)], name)  # type: ignore
            w = (
                layer_mod.base if isinstance(layer_mod, LoRALinear) else layer_mod
            ).weight
            a = f.get_tensor(key).to(w.device, torch.float32)
            b = f.get_tensor(key[:-1] + "B").to(w.device, torch.float32)
            w.copy_((w.float() + strength * (b @ a)).to(w.dtype))
