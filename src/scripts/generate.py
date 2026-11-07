import argparse
import os
import time

import numpy as np
import torch
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from PIL import Image

from ..lora import merge_lora_file
from ..model import AnimaFlat, load
from ..sample import NOISE_SCALES, SAMPLERS, decode, noise
from ..vae import VAE
from .verify import convert_copy, encode_prompt, pad_to

# Defaults of the reference ComfyUI workflow shipped with the model.
PROMPT = (
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top and denim "
    "shorts is standing outdoors. She's holding a rectangular sign out in front of her that reads "
    '"ANIMA". She\'s looking at the viewer with a smile. The background features some trees and '
    "blue sky with clouds."
)
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"


def add_args(ap: argparse.ArgumentParser) -> None:
    """Add the model and sampling options shared by the generation scripts.

    Args:
        ap: Parser to extend.
    """
    ap.add_argument(
        "--repo",
        default="circlestone-labs/Anima-Base-v1.0-Diffusers",
        help="Diffusers repo providing the text encoder, tokenizers and VAE.",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="anima-flat checkpoint; if omitted, the repo's transformer is converted.",
    )
    ap.add_argument("--lora", default=None, help="anima-flat-lora-1 adapter to merge.")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--qk-eps", type=float, default=1e-5)
    ap.add_argument(
        "--trim-context",
        action="store_true",
        help="Skip padded text tokens in cross-attention (CUDA with FA4 only).",
    )
    ap.add_argument(
        "--fp8",
        action="store_true",
        help="FP8 qkv and cross-attention query projections (Hopper or newer; inference only).",
    )
    ap.add_argument(
        "--diffusers-vae",
        action="store_true",
        help="Decode with the diffusers VAE instead of the fused single-frame decoder.",
    )
    ap.add_argument(
        "--compile-vae",
        action="store_true",
        help="Compile the diffusers VAE decoder and use it (implies --diffusers-vae); slower first image and per new resolution.",
    )
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--width", type=int, default=1024, help="Pixels, multiple of 16.")
    ap.add_argument("--height", type=int, default=1024, help="Pixels, multiple of 16.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument(
        "--cfg", type=float, default=4.0, help="Guidance scale; 1 disables guidance."
    )
    ap.add_argument("--shift", type=float, default=3.0, help="Noise schedule shift.")
    ap.add_argument("--sampler", choices=SAMPLERS, default="euler")
    ap.add_argument(
        "--max-stage",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Highest solver order of er_sde.",
    )
    ap.add_argument(
        "--s-noise",
        type=float,
        default=1.0,
        help="Multiplier on the noise er_sde injects; below 1 shrinks the result.",
    )
    ap.add_argument(
        "--noise-scale",
        choices=NOISE_SCALES,
        default="er",
        help="er_sde variant: er (stochastic) or ode (noise-free multistep solver).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed of the first image.")


def check_size(args: argparse.Namespace) -> None:
    """Validate the requested image size.

    Args:
        args: Parsed options with `width` and `height`.

    Raises:
        SystemExit: If a side is not a positive multiple of 16.
    """
    for name in ("width", "height"):
        v = getattr(args, name)
        if v < 16 or v % 16:
            raise SystemExit(f"--{name} must be a positive multiple of 16, got {v}")


def load_models(args: argparse.Namespace) -> tuple[ModularPipeline, AnimaFlat]:
    """Load the text encoder, VAE and the flat model described by the options.

    Args:
        args: Options from `add_args`.

    Returns:
        The diffusers pipeline (for text encoding and decoding) and the flat model in eval mode.

    Raises:
        ValueError: If `args.device` is not a CUDA device.
    """
    if not args.device.startswith("cuda"):
        raise ValueError("the flat model needs a CUDA device")
    dtype = getattr(torch, args.dtype)
    pipe = ModularPipeline.from_pretrained(args.repo)
    pipe.load_components(dtype=dtype)
    pipe.to(args.device)
    if args.checkpoint:
        model = load(args.checkpoint, device=args.device, dtype=dtype)
    else:
        model = convert_copy(pipe.transformer.eval(), dtype, args.device, args.qk_eps)
    if args.lora:
        merge_lora_file(model, args.lora, args.lora_strength)
    model.trim_context = args.trim_context
    if args.fp8:
        model.enable_fp8()
    if args.compile_vae:
        pipe.vae.decoder = torch.compile(pipe.vae.decoder)
    elif not args.diffusers_vae:
        pipe.vae.fast = VAE(pipe.vae).to(args.device, dtype).eval()
    return pipe, model  # type: ignore


@torch.no_grad()
def embed(
    pipe: ModularPipeline, prompt: str, negative: str, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a prompt pair into model context.

    Args:
        pipe: Pipeline providing the tokenizers and text encoders.
        prompt: Positive prompt.
        negative: Negative prompt.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Positive and negative embeddings, zero-padded to the same length, shape `[1, S, text_dim]`.
    """
    cond = encode_prompt(pipe, prompt, device, dtype)
    unc = encode_prompt(pipe, negative, device, dtype)
    n = max(cond.shape[1], unc.shape[1])
    return pad_to(cond, n), pad_to(unc, n)


@torch.no_grad()
def denoise(
    model: AnimaFlat,
    cond: torch.Tensor,
    unc: torch.Tensor,
    args: argparse.Namespace,
    seeds: list[int],
) -> torch.Tensor:
    """Denoise one batch of latents, one per seed.

    Args:
        model: The flat model.
        cond: Positive embeddings from `embed`.
        unc: Negative embeddings from `embed`.
        args: Options with `width`, `height`, `steps`, `cfg`, `shift`, `sampler`, `max_stage`,
            `s_noise`, `noise_scale`, `device`.
        seeds: One noise seed per image; the result for a seed does not depend on the batch.

    Returns:
        Float32 latents, shape `[B, height/8, width/8, C]`.
    """
    B = len(seeds)
    h, w, C = args.height // 8, args.width // 8, model.cfg.in_ch
    x = torch.cat([noise(1, h, w, C, s, args.device) for s in seeds])
    return model.denoise(
        cond.expand(B, -1, -1).contiguous(),
        unc.expand(B, -1, -1).contiguous(),
        x,
        args.steps,
        scale=args.cfg,
        shift=args.shift,
        sampler=args.sampler,
        max_stage=args.max_stage,
        s_noise=args.s_noise,
        seeds=seeds,
        noise_scale=args.noise_scale,
    )


def decode_latents(pipe: ModularPipeline, x: torch.Tensor) -> torch.Tensor:
    """Decode latents with the fused VAE decoder, or the diffusers VAE with `--diffusers-vae`.

    Args:
        pipe: Pipeline providing the VAE.
        x: Latents from `denoise`.

    Returns:
        Images `[B, 3, H, W]` in `[-1, 1]`, left on the device.
    """
    vae = pipe.vae
    if hasattr(vae, "fast"):
        return vae.fast.decode(x)
    return decode(vae, vae.config.latents_mean, vae.config.latents_std, x)


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert decoded images to 8-bit arrays.

    Args:
        img: Images `[B, 3, H, W]` in `[-1, 1]`.

    Returns:
        `uint8` array of shape `[B, H, W, 3]` on the CPU.
    """
    x = (img.float() / 2 + 0.5).clamp(0, 1)
    return (x * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def main(argv: list[str] | None = None) -> int:
    """Generate images with the flat model and save them as PNG files.

    Files are named `<seed>.png`; image `i` of a run uses seed `--seed + i`.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on success.
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    add_args(ap)
    ap.add_argument("--count", type=int, default=1, help="Number of images.")
    ap.add_argument("--batch", type=int, default=1, help="Images denoised together.")
    ap.add_argument("--out", default="generate_out")
    args = ap.parse_args(argv)
    check_size(args)
    if args.count < 1 or args.batch < 1:
        raise SystemExit("--count and --batch must be at least 1")

    dtype = getattr(torch, args.dtype)
    pipe, model = load_models(args)
    cond, unc = embed(pipe, args.prompt, args.negative, args.device, dtype)
    os.makedirs(args.out, exist_ok=True)
    seeds = [args.seed + i for i in range(args.count)]
    t0 = time.perf_counter()
    for i in range(0, len(seeds), args.batch):
        chunk = seeds[i : i + args.batch]
        img = decode_latents(pipe, denoise(model, cond, unc, args, chunk))
        for seed, arr in zip(chunk, to_uint8(img), strict=True):
            path = os.path.join(args.out, f"{seed}.png")
            Image.fromarray(arr).save(path)
            print(path)
    dt = time.perf_counter() - t0
    print(
        f"{args.count} image(s) in {dt:.1f} s ({dt / args.count:.2f} s each, first call includes warmup)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
