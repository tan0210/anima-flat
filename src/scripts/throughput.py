import argparse
import json
import time

import numpy as np
import torch

from .generate import add_args, check_size, decode_latents, denoise, embed, load_models


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device: str):
    """Run `fn` and time it.

    Args:
        fn: Callable taking no arguments.
        device: Device string; the device is synchronized before and after.

    Returns:
        The result of `fn` and its wall time in milliseconds.
    """
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return out, (time.perf_counter() - t0) * 1e3


def run_batch(pipe, model, args, dtype: torch.dtype, batch: int, cond, unc) -> dict:
    """Generate one batch and time each stage; nothing is copied off the GPU.

    Args:
        pipe: Pipeline providing the text encoders and VAE.
        model: The flat model.
        args: Generation options.
        dtype: Compute dtype.
        batch: Number of images.
        cond: Cached positive embeddings, or None to encode the prompts in the timed region.
        unc: Cached negative embeddings, or None.

    Returns:
        Stage times in milliseconds: `encode`, `denoise`, `decode`, `total`.
    """
    seeds = [args.seed + i for i in range(batch)]
    enc = 0.0
    if cond is None:
        (cond, unc), enc = timed(
            lambda: embed(pipe, args.prompt, args.negative, args.device, dtype),
            args.device,
        )
    x, den = timed(lambda: denoise(model, cond, unc, args, seeds), args.device)
    _, dec = timed(lambda: decode_latents(pipe, x), args.device)
    return {"encode": enc, "denoise": den, "decode": dec, "total": enc + den + dec}


def run_diffusers(pipe, args, channels: int) -> float:
    """Time one image through the stock diffusers pipeline.

    Args:
        pipe: The loaded pipeline.
        args: Generation options.
        channels: Latent channels.

    Returns:
        Wall time in milliseconds, including its conversion to a PIL image in memory. The
        pipeline uses its own scheduler shift, not `--shift`.
    """
    if getattr(pipe, "guider", None) is not None:
        pipe.guider.guidance_scale = args.cfg
    g = torch.Generator().manual_seed(args.seed)
    lat = torch.randn(1, channels, 1, args.height // 8, args.width // 8, generator=g)

    def call():
        return pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            latents=lat.to(args.device).clone(),
        ).images[0]

    _, ms = timed(call, args.device)
    return ms


def main(argv: list[str] | None = None) -> int:
    """Measure end-to-end generation throughput and cost.

    Runs the full path (text encoding, denoising, VAE decoding) at the given settings and
    reports images per second and cost per image for each batch size. Images stay on the GPU
    and are never written to disk. The first `--warmup` runs are discarded because they include
    kernel compilation.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on success.
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    add_args(ap)
    ap.add_argument(
        "--batches", type=int, nargs="+", default=[1], help="Batch sizes to test."
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument(
        "--price", type=float, default=3.36, help="GPU price in USD per hour."
    )
    ap.add_argument(
        "--cache-text",
        action="store_true",
        help="Encode the prompts once instead of inside every timed run.",
    )
    ap.add_argument(
        "--json",
        default=None,
        help="Write the results to this file for `compare speed`.",
    )
    ap.add_argument(
        "--diffusers",
        action="store_true",
        help="Also time the stock diffusers pipeline at batch 1.",
    )
    args = ap.parse_args(argv)
    check_size(args)

    dtype = getattr(torch, args.dtype)
    pipe, model = load_models(args)
    cached = (
        embed(pipe, args.prompt, args.negative, args.device, dtype)
        if args.cache_text
        else (None, None)
    )

    rows = []
    for b in args.batches:
        for _ in range(args.warmup):
            run_batch(pipe, model, args, dtype, b, *cached)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        runs = [
            run_batch(pipe, model, args, dtype, b, *cached) for _ in range(args.iters)
        ]
        peak = (
            (torch.cuda.max_memory_allocated() - base) / 2**30  # type: ignore
            if args.device.startswith("cuda")
            else float("nan")
        )
        med = {k: float(np.median([r[k] for r in runs])) for k in runs[0]}
        rows.append((b, med, peak))

    hf_ms = None
    if args.diffusers:
        for _ in range(args.warmup):
            run_diffusers(pipe, args, model.cfg.in_ch)
        hf_ms = float(
            np.median(
                [run_diffusers(pipe, args, model.cfg.in_ch) for _ in range(args.iters)]
            )
        )

    per_sec = args.price / 3600
    print("## Settings\n")
    print(
        f"- size: {args.width}x{args.height}, steps: {args.steps}, cfg: {args.cfg}, shift: {args.shift}"
    )
    print(
        f"- dtype: {args.dtype}, device: {args.device}"
        + (
            f" ({torch.cuda.get_device_name()})"
            if args.device.startswith("cuda")
            else ""
        )
    )
    print(
        f"- warmup runs: {args.warmup}, timed runs: {args.iters}, price: ${args.price}/h"
    )
    print(f"- text encoding {'excluded (cached)' if args.cache_text else 'included'}\n")
    print("## Throughput\n")
    print(
        "| batch | ms/image | images/s | USD/image | USD/1000 images | encode ms | denoise ms | decode ms | peak GiB |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for b, m, peak in rows:
        ms_img = m["total"] / b
        cost = per_sec * ms_img / 1e3
        print(
            f"| {b} | {ms_img:.1f} | {1e3 / ms_img:.2f} | {cost:.5f} | {cost * 1000:.2f} | "
            f"{m['encode']:.1f} | {m['denoise']:.1f} | {m['decode']:.1f} | {peak:.2f} |"
        )
    if args.json:
        out = {
            "fp8": args.fp8,
            "diffusers_ms_per_image": hf_ms,
            "settings": {
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "cfg": args.cfg,
                "shift": args.shift,
                "sampler": args.sampler,
                "price": args.price,
                "cache_text": args.cache_text,
            },
            "rows": [
                {
                    "batch": b,
                    "ms_per_image": m["total"] / b,
                    "images_per_s": 1e3 / (m["total"] / b),
                    "usd_per_1000": per_sec * (m["total"] / b) / 1e3 * 1000,
                    "peak_gib": peak,
                }
                for b, m, peak in rows
            ],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    if hf_ms is not None:
        cost = per_sec * hf_ms / 1e3
        print(
            f"\ndiffusers pipeline, batch 1: {hf_ms:.1f} ms/image, {1e3 / hf_ms:.2f} images/s, "
            f"${cost:.5f}/image (includes conversion to a PIL image)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
