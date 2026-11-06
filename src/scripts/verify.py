import argparse
import os
import tempfile

import numpy as np
import torch
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from PIL import Image

from ..attention import use_fa4
from ..model import AnimaFlat, load
from ..sample import sigmas
from .convert import (
    config_from_diffusers,
    convert_checkpoint,
    from_diffusers,
    to_diffusers,
)

NEGATIVE = "worst quality, low quality"
RESULTS: list[tuple[str, bool]] = []


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error of `a` against the reference `b`.

    Args:
        a: Tensor under test.
        b: Reference tensor of the same shape.

    Returns:
        `||a - b|| / ||b||` computed in float32.
    """
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    RESULTS.append((name, ok))
    return ok


def check_schedule(pipe, steps: int) -> None:
    sch = pipe.scheduler
    sch.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), device="cpu")
    ref = sch.sigmas.float()
    ours = sigmas(steps, sch.config.shift)
    err = (ref - ours).abs().max().item()
    check("sigma schedule matches scheduler", err < 1e-5, f"max abs diff {err:.2e}")


def check_weights(tr) -> None:
    sd = {k: v.detach().cpu() for k, v in tr.state_dict().items()}
    cfg = config_from_diffusers(sd, dict(tr.config))
    back = to_diffusers(from_diffusers(sd, cfg), cfg)
    dropped = sorted(set(sd) - set(back))
    check("no weights dropped by conversion", not dropped, f"dropped: {dropped[:6]}")
    bad = []
    for k in sd.keys() & back.keys():
        if k == "patch_embed.proj.weight":
            D, C = sd[k].shape[0], cfg.in_ch
            a = sd[k].view(D, C + 1, 2, 2)
            b = back[k].view(D, C + 1, 2, 2)
            if not torch.equal(a[:, :C], b[:, :C]):
                bad.append(k)
            print(
                f"       padding-mask channel |w|max = {a[:, C].abs().max().item():.3e}"
            )
        elif not torch.equal(sd[k], back[k]):
            bad.append(k)
    check("weights round-trip exactly", not bad, f"mismatched: {bad[:6]}")


def convert_copy(tr, dtype, device, qk_eps) -> AnimaFlat:
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "diffusers")
        tr.save_pretrained(src)
        path = os.path.join(d, "flat.safetensors")
        convert_checkpoint(src, path, qk_eps=qk_eps)
        return load(path, device=device, dtype=dtype)


@torch.no_grad()
def check_forward(
    tr, model: AnimaFlat, dtype, device, size, ctx_len, tol, seed=0
) -> None:
    H = W = size // 8
    C = model.cfg.in_ch
    B = 2
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, C, 1, H, W, generator=g)
    t = (torch.rand(B, generator=g) * 0.9 + 0.05).to(dtype).float()
    ctx = torch.zeros(B, ctx_len, model.cfg.text_dim)
    real = max(1, ctx_len // 6)
    ctx[:, :real] = torch.randn(B, real, model.cfg.text_dim, generator=g)
    x_d, ctx_d = x.to(device, dtype), ctx.to(device, dtype)
    pad = torch.zeros(1, 1, size, size, device=device, dtype=dtype)

    acts_ref, acts_our, hooks = {}, {}, []
    for i, b in enumerate(tr.transformer_blocks):
        hooks.append(
            b.register_forward_hook(
                lambda _m, _i, o, i=i: acts_ref.__setitem__(
                    i, (o[0] if isinstance(o, tuple) else o).float()
                )
            )
        )
    ref = tr(
        hidden_states=x_d,
        timestep=t.to(device, dtype),
        encoder_hidden_states=ctx_d,
        padding_mask=pad,
        return_dict=False,
    )[0]
    ref = ref[:, :, 0].permute(0, 2, 3, 1)  # [B, H, W, C]
    for h in hooks:
        h.remove()

    # A block returns its updated residual stream except for the last one, whose residual is
    # added by the output layer, so that one is covered by the final output check.
    run_block = model._fused_block

    def record(l, *args):
        out = run_block(l, *args)
        if l + 1 < model.cfg.layers:
            acts_our[l] = out[1].float()
        return out

    xo = x[:, :, 0].permute(0, 2, 3, 1).contiguous().to(device, dtype)
    model.trim_context = False
    model._fused_block = record  # type: ignore
    try:
        out = model(xo, t.to(device), model.encode_context(ctx_d))
    finally:
        del model._fused_block

    errs = [rel_err(acts_our[i], acts_ref[i]) for i in sorted(acts_our)]
    print("       per-block relative error: " + " ".join(f"{e:.1e}" for e in errs))
    check("per-block outputs match", max(errs) < tol, f"max {max(errs):.2e}")
    check(
        "output matches diffusers",
        rel_err(out, ref) < tol,
        f"rel err {rel_err(out, ref):.2e}",
    )
    if use_fa4(xo):
        model.trim_context = True
        out_t = model(xo, t.to(device), model.encode_context(ctx_d))
        model.trim_context = False
        check(
            "trimmed-context output matches diffusers",
            rel_err(out_t, ref) < tol,
            f"rel err {rel_err(out_t, ref):.2e}",
        )


@torch.no_grad()
def encode_prompt(pipe, prompt: str, device, dtype) -> torch.Tensor:
    qwen = pipe.tokenizer(
        [prompt],
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    ids, mask = qwen.input_ids.to(device), qwen.attention_mask.to(device)
    h = pipe.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    h = h * mask.to(h).unsqueeze(-1)
    t5 = pipe.t5_tokenizer(
        [prompt],
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    out = pipe.text_conditioner(
        source_hidden_states=h.to(pipe.text_conditioner.dtype),
        target_input_ids=t5.input_ids.to(device),
        target_attention_mask=t5.attention_mask.to(device),
        source_attention_mask=mask,
    )
    return out.to(dtype)


def pad_to(t: torch.Tensor, n: int) -> torch.Tensor:
    return torch.nn.functional.pad(t, (0, 0, 0, n - t.shape[1]))


@torch.no_grad()
def decode_ours(pipe, x: torch.Tensor) -> np.ndarray:
    vae = pipe.vae
    mean = torch.tensor(vae.config.latents_mean, device=x.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=x.device).view(1, -1, 1, 1, 1)
    z = x.permute(0, 3, 1, 2).unsqueeze(2) * std + mean
    img = vae.decode(z.to(vae.dtype), return_dict=False)[0][:, :, 0]
    img = (img.float() / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (img * 255).round().astype(np.uint8)


@torch.no_grad()
def check_generation(pipe, model, args, dtype, device) -> None:
    os.makedirs(args.out, exist_ok=True)
    size, C = args.size, model.cfg.in_ch
    g = torch.Generator().manual_seed(args.seed)
    lat = torch.randn(1, C, 1, size // 8, size // 8, generator=g)

    ref_img = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        height=size,
        width=size,
        latents=lat.to(device).clone(),
    ).images[0]
    ref = np.asarray(ref_img.convert("RGB"))

    cond = encode_prompt(pipe, args.prompt, device, dtype)
    unc = encode_prompt(pipe, args.negative, device, dtype)
    n = max(cond.shape[1], unc.shape[1])
    cond, unc = pad_to(cond, n), pad_to(unc, n)
    x = lat[:, :, 0].permute(0, 2, 3, 1).contiguous().to(device)
    out = model.denoise(
        cond, unc, x, args.steps, scale=4.0, shift=pipe.scheduler.config.shift
    )
    ours = decode_ours(pipe, out)

    Image.fromarray(ref).save(os.path.join(args.out, "reference.png"))
    Image.fromarray(ours).save(os.path.join(args.out, "ours.png"))
    Image.fromarray(np.concatenate([ref, ours], axis=1)).save(
        os.path.join(args.out, "side_by_side.png")
    )
    diff = ref.astype(np.float32) - ours.astype(np.float32)
    mse = float((diff**2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    print(
        f"       mean abs pixel diff {np.abs(diff).mean():.2f}, PSNR {psnr:.1f} dB; "
        f"images in {args.out}/"
    )
    check(
        "generated image resembles diffusers' image",
        psnr >= args.min_psnr,
        f"PSNR {psnr:.1f} dB (min {args.min_psnr})",
    )


def main(argv: list[str] | None = None) -> int:
    """Compare a converted model with the diffusers implementation.

    Converts a copy of the diffusers transformer to the flat layout, then checks the weights,
    the per-block and final outputs for the same inputs, and the noise schedule, and finally
    generates one image with both implementations from the same noise and prompt.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--repo", default="circlestone-labs/Anima-Base-v1.0-Diffusers")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--prompt", default="masterpiece, best quality, 1girl, solo, city lights"
    )
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument(
        "--size", type=int, default=512, help="Image side in pixels (multiple of 16)."
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--qk-eps", type=float, default=1e-5)
    ap.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Max relative error for forward checks (default 5e-2 bf16, 1e-3 fp32).",
    )
    ap.add_argument("--min-psnr", type=float, default=25.0)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--out", default="verify_out")
    args = ap.parse_args(argv)
    if not args.device.startswith("cuda"):
        ap.error("the flat model needs a CUDA device")

    dtype = getattr(torch, args.dtype)
    tol = (
        args.tol
        if args.tol is not None
        else (5e-2 if dtype == torch.bfloat16 else 1e-3)
    )

    pipe = ModularPipeline.from_pretrained(args.repo)
    pipe.load_components(dtype=dtype)
    pipe.to(args.device)
    tr = pipe.transformer.eval()

    check_schedule(pipe, args.steps)
    check_weights(tr)
    model = convert_copy(tr, dtype, args.device, args.qk_eps)
    check_forward(tr, model, dtype, args.device, args.size, args.ctx_len, tol)
    if not args.skip_generate:
        check_generation(pipe, model, args, dtype, args.device)

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
