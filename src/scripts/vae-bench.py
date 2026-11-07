import argparse

import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)

from ..vae import VAE


def measure(fn, iters: int = 10) -> tuple[float, float]:
    """Time a CUDA callable.

    Args:
        fn: Callable to run.
        iters: Timed iterations after two warmups.

    Returns:
        `(milliseconds per call, peak allocated GiB during the timed calls)`.
    """
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare the fast VAE decoder with diffusers."
    )
    ap.add_argument("--repo", default="circlestone-labs/Anima-Base-v1.0-Diffusers")
    ap.add_argument("--subfolder", default="vae")
    ap.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Image side for the inference comparison.",
    )
    ap.add_argument(
        "--crop", type=int, default=256, help="Image side for the training comparison."
    )
    ap.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    vae = (
        AutoencoderKLQwenImage.from_pretrained( # type: ignore
            args.repo, subfolder=args.subfolder, torch_dtype=dtype
        )
        .cuda()
        .eval()
    )
    fast = VAE(vae).cuda().eval()
    C = vae.config.z_dim

    def latents(side: int) -> torch.Tensor:
        g = torch.Generator("cuda").manual_seed(0)
        return torch.randn(
            1, C, side // 8, side // 8, device="cuda", dtype=dtype, generator=g
        )

    def ref_decode(z):
        return vae.decode(z.unsqueeze(2), return_dict=False)[0][:, :, 0]

    rows = []
    z = latents(args.size)
    with torch.no_grad():
        a, b = ref_decode(z).float().clamp(-1, 1), fast(z).float().clamp(-1, 1)
        err = (a - b).abs().max().item()
        t_ref, m_ref = measure(lambda: ref_decode(z))
        t_fast, m_fast = measure(lambda: fast(z))
    rows.append((f"Decode {args.size}px", t_ref, m_ref, t_fast, m_fast))

    zc = latents(args.crop).requires_grad_()

    def train_ref():
        ref_decode(zc).float().square().mean().backward()
        vae.zero_grad(set_to_none=True)

    def train_fast():
        fast(zc).float().square().mean().backward()
        fast.zero_grad(set_to_none=True)

    for p in vae.decoder.parameters():
        p.requires_grad_(True)
    for p in vae.post_quant_conv.parameters():
        p.requires_grad_(True)
    t_ref, m_ref = measure(train_ref)
    t_fast, m_fast = measure(train_fast)
    rows.append(
        (f"Forward and backward {args.crop}px crop", t_ref, m_ref, t_fast, m_fast)
    )

    print(
        f"VAE `{args.repo}`, {args.dtype}, maximum absolute pixel difference {err:.4f} (images in [-1, 1])\n"
    )
    print(
        "| Pass | diffusers (ms) | diffusers peak (GiB) | Fast (ms) | Fast peak (GiB) | Speedup |"
    )
    print("|---|---|---|---|---|---|")
    for name, tr, mr, tf, mf in rows:
        print(
            f"| {name} | {tr:.1f} | {mr:.2f} | {tf:.1f} | {mf:.2f} | {tr / tf:.2f}x |"
        )


if __name__ == "__main__":
    main()
