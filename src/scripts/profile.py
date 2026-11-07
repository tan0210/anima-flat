import argparse
import json
import os
import platform
import time
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline

from ..attention import use_fa4
from ..kernels.gemm import have_quack
from ..lora import add_lora, remove_lora
from ..train import flow_loss
from ..vae import VAE
from .verify import NEGATIVE, convert_copy, decode_ours, encode_prompt, pad_to

ROWS: list[dict] = []

CATEGORIES = (
    "attention",
    "gemm",
    "our triton",
    "norm/elementwise",
    "copy/layout",
    "other triton",
    "other",
)
OUR_KERNELS = (
    "_qkv_norm_rope",
    "_head_norm",
    "_residual_adaln",
    "_adaln",
    "_ctx_scale",
    "_rowquant",
)


def classify(name: str) -> str:
    """Assign a GPU kernel name to a coarse category.

    Args:
        name: Kernel name as reported by the profiler.

    Returns:
        One of `CATEGORIES`.
    """
    n = name.lower()
    if any(s in n for s in ("flash", "fmha", "sdpa", "attn", "attention")):
        return "attention"
    if any(n.startswith(k) for k in OUR_KERNELS):
        return "our triton"
    if any(
        s in n for s in ("gemm", "cutlass", "nvjet", "xmma", "cublas", "quack", "gemv")
    ):
        return "gemm"
    if any(
        s in n
        for s in (
            "memcpy",
            "memset",
            "copy",
            "transpose",
            "permute",
            "catarray",
            "index",
        )
    ):
        return "copy/layout"
    if any(
        s in n
        for s in (
            "elementwise",
            "layer_norm",
            "rms",
            "norm",
            "reduce",
            "softmax",
            "sigmoid",
            "gelu",
        )
    ):
        return "norm/elementwise"
    if "triton" in n:
        return "other triton"
    return "other"


def profiled(args, size: int) -> bool:
    """Tell whether kernel profiling is requested at this size.

    Args:
        args: Parsed options with `skip_profile`, `profile_size` and `sizes`.
        size: Image size being benchmarked.

    Returns:
        True if this size gets a kernel breakdown.
    """
    target = args.profile_size if args.profile_size else args.sizes[0]
    return not args.skip_profile and size == target


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def bench(fn: Callable[[], object], warmup: int, iters: int, device: str) -> dict:
    """Time a callable.

    Args:
        fn: Callable to run; its result is discarded.
        warmup: Untimed calls beforehand.
        iters: Timed calls.
        device: Device string; CUDA events are used on CUDA, wall clock otherwise.

    Returns:
        Dict with `median`, `mean` and `min` in milliseconds.
    """
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        if device.startswith("cuda"):
            s, e = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            s.record()
            fn()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times)
    return {
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "min": float(a.min()),
    }


def peak_gb(fn: Callable[[], object], device: str) -> float:
    """Peak extra GPU memory allocated while running `fn`, in GiB (NaN on CPU)."""
    if not device.startswith("cuda"):
        return float("nan")
    sync(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    sync(device)
    return (torch.cuda.max_memory_allocated() - base) / 2**30


def weights_gb(module: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in module.parameters()) / 2**30


def forward_flops(cfg, size: int, batch: int, ctx_len: int) -> float:
    """Approximate FLOPs of one transformer forward pass (matmuls and attention only)."""
    N, D, M, S, T = (size // 16) ** 2, cfg.dim, cfg.mlp_dim, ctx_len, cfg.text_dim
    per = (
        2 * N * D * 3 * D  # qkv
        + 2 * N * D * D  # o1
        + 4 * N * N * D  # self-attention
        + 2 * N * D * D  # xq
        + 2 * S * T * 2 * D  # xkv
        + 4 * N * S * D  # cross-attention
        + 2 * N * D * D  # o2
        + 4 * N * D * M  # mlp up + down
    )
    return float(batch * cfg.layers * per)


def add_row(
    section: str,
    name: str,
    size: int,
    stats: dict | None,
    ref_ms: float | None = None,
    flops: float | None = None,
    mem: float | None = None,
    note: str = "",
) -> None:
    ms = stats["median"] if stats else None
    ROWS.append(
        {
            "section": section,
            "name": name,
            "size": size,
            "ms": ms,
            "speedup": (ref_ms / ms) if (ms and ref_ms) else None,
            "tflops": (flops / (ms / 1e3) / 1e12) if (ms and flops) else None,
            "mem_gb": mem,
            "note": note,
        }
    )


def make_inputs(cfg, size, batch, ctx_len, dtype, device, seed=0):
    H = W = size // 8
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, cfg.in_ch, 1, H, W, generator=g)
    ctx = torch.zeros(batch, ctx_len, cfg.text_dim)
    real = min(ctx_len, 77)
    ctx[:, :real] = torch.randn(batch, real, cfg.text_dim, generator=g)
    t = torch.full((batch,), 0.5)
    pad = torch.zeros(1, 1, size, size)
    return (
        x.to(device, dtype),
        t.to(device, dtype),
        ctx.to(device, dtype),
        pad.to(device, dtype),
    )


def kernel_table(
    prof, wall_ms: float, iters: int, device: str, label: str, top: int
) -> dict:
    cuda_kernels = []
    for e in prof.key_averages():
        if e.device_type.name == "CUDA":
            t = getattr(e, "self_device_time_total", None)
            if t is None:
                t = e.self_cuda_time_total
            cuda_kernels.append((e.key, e.count, t))
    if not cuda_kernels:
        ops = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:top]
        print(f"\n### {label}: no CUDA kernels recorded, top CPU ops\n")
        print("| op | calls/iter | ms/iter |\n|---|---|---|")
        for e in ops:
            print(
                f"| `{e.key[:60]}` | {e.count / iters:.1f} | {e.self_cpu_time_total / 1e3 / iters:.3f} |"
            )
        return {}
    total_us = sum(t for _, _, t in cuda_kernels)
    launches = sum(c for _, c, _ in cuda_kernels) / iters
    by_cat: dict[str, float] = defaultdict(float)
    for k, _, t in cuda_kernels:
        by_cat[classify(k)] += t
    busy_ms = total_us / 1e3 / iters
    print(f"\n### {label}\n")
    print(
        f"{wall_ms:.2f} ms per call, {busy_ms:.2f} ms of GPU kernel time "
        f"({100 * busy_ms / wall_ms:.0f}% busy), {launches:.0f} kernel launches per call.\n"
    )
    print("| category | ms | share |\n|---|---|---|")
    for c in CATEGORIES:
        if by_cat.get(c):
            print(
                f"| {c} | {by_cat[c] / 1e3 / iters:.3f} | {100 * by_cat[c] / total_us:.1f}% |"
            )
    print(
        "\n| kernel | calls | ms | avg us | share | category |\n|---|---|---|---|---|---|"
    )
    for k, c, t in sorted(cuda_kernels, key=lambda r: -r[2])[:top]:
        print(
            f"| `{k[:64]}` | {c / iters:.1f} | {t / 1e3 / iters:.3f} | {t / c:.1f} | "
            f"{100 * t / total_us:.1f}% | {classify(k)} |"
        )
    return {
        "wall_ms": wall_ms,
        "busy_ms": busy_ms,
        "launches": launches,
        "by_cat": {c: v / 1e3 / iters for c, v in by_cat.items()},
    }


def profile_call(
    fn, label: str, out_dir: str, device: str, active: int, top: int, wall_ms: float
) -> dict:
    for _ in range(3):
        fn()
    sync(device)
    acts = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=acts) as prof:
        for _ in range(active):
            fn()
        sync(device)
    os.makedirs(out_dir, exist_ok=True)
    prof.export_chrome_trace(os.path.join(out_dir, f"{label.replace(' ', '_')}.json"))
    return kernel_table(prof, wall_ms, active, device, label, top)


def hints(
    label: str, info: dict, peak_tflops: float, tflops: float | None
) -> list[str]:
    if not info:
        return []
    out = []
    busy = 100 * info["busy_ms"] / info["wall_ms"]
    if busy < 85:
        out.append(
            f"{label}: GPU only {busy:.0f}% busy; CPU launch overhead matters "
            f"({info['launches']:.0f} launches/iter). Fewer, fused launches would help."
        )
    total = sum(info["by_cat"].values()) or 1.0
    ew = (
        info["by_cat"].get("norm/elementwise", 0) + info["by_cat"].get("copy/layout", 0)
    ) / total
    if ew > 0.12:
        out.append(
            f"{label}: {100 * ew:.0f}% of kernel time is unfused elementwise/copy work; "
            "check the top-kernels table for what is still unfused."
        )
    gemm = info["by_cat"].get("gemm", 0) / total
    if tflops and peak_tflops and gemm > 0.5 and tflops < 0.4 * peak_tflops:
        out.append(
            f"{label}: GEMMs dominate but overall throughput is {tflops:.0f} TFLOP/s "
            f"({100 * tflops / peak_tflops:.0f}% of peak); check GEMM tile choices and shapes."
        )
    attn = info["by_cat"].get("attention", 0) / total
    if attn > 0.35:
        out.append(
            f"{label}: attention is {100 * attn:.0f}% of kernel time; "
            "confirm FA4 is active and not the SDPA fallback."
        )
    return out


def run_forward(pipe, tr, model, args, dtype, device, size, flops):
    cfg = model.cfg
    x, t, ctx, pad = make_inputs(cfg, size, args.batch, args.ctx_len, dtype, device)
    xo = x[:, :, 0].permute(0, 2, 3, 1).contiguous()
    t32 = t.float()

    def hf():
        return tr(
            hidden_states=x,
            timestep=t,
            encoder_hidden_states=ctx,
            padding_mask=pad,
            return_dict=False,
        )

    model.trim_context = False
    cached = model.encode_context(ctx)

    def ours():
        return model(xo, t32, cached)

    results = {}
    with torch.no_grad():
        s_hf = bench(hf, args.warmup, args.iters, device)
        add_row(
            "forward",
            "diffusers",
            size,
            s_hf,
            s_hf["median"],
            flops,
            peak_gb(hf, device),
        )
        s_o = bench(ours, args.warmup, args.iters, device)
        add_row(
            "forward",
            "ours fused",
            size,
            s_o,
            s_hf["median"],
            flops,
            peak_gb(ours, device),
        )
        if args.fp8:
            model.enable_fp8()
            s_f = bench(ours, args.warmup, args.iters, device)
            add_row(
                "forward",
                "ours fused (fp8)",
                size,
                s_f,
                s_hf["median"],
                flops,
                peak_gb(ours, device),
            )
            model.disable_fp8()
        if profiled(args, size):
            ih = profile_call(
                hf,
                f"diffusers forward {size}",
                args.out,
                device,
                args.profile_iters,
                args.top,
                s_hf["median"],
            )
            io = profile_call(
                ours,
                f"ours forward {size}",
                args.out,
                device,
                args.profile_iters,
                args.top,
                s_o["median"],
            )
            results["hf"], results["ours"] = ih, io
            tf_o = flops / (s_o["median"] / 1e3) / 1e12
            tf_h = flops / (s_hf["median"] / 1e3) / 1e12
            lines = hints("ours", io, args.peak_tflops, tf_o) + hints(
                "diffusers", ih, args.peak_tflops, tf_h
            )
            if args.fp8:
                model.enable_fp8()
                iq = profile_call(
                    ours,
                    f"ours fp8 forward {size}",
                    args.out,
                    device,
                    args.profile_iters,
                    args.top,
                    s_f["median"],  # type: ignore
                )
                model.disable_fp8()
                results["ours_fp8"] = iq
                tf_q = flops / (s_f["median"] / 1e3) / 1e12  # type: ignore
                lines += hints("ours fp8", iq, args.peak_tflops, tf_q)
            if lines:
                print("\n### Hints\n")
                for line in lines:
                    print("-", line)
    return results


def run_e2e(pipe, model, args, dtype, device, size):
    g = torch.Generator().manual_seed(0)
    lat = torch.randn(1, model.cfg.in_ch, 1, size // 8, size // 8, generator=g)
    shift = pipe.scheduler.config.shift

    def hf():
        return pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            height=size,
            width=size,
            latents=lat.to(device).clone(),
        ).images[0]

    with torch.no_grad():
        cond = encode_prompt(pipe, args.prompt, device, dtype)
        unc = encode_prompt(pipe, args.negative, device, dtype)
        n = max(cond.shape[1], unc.shape[1])
        cond_p, unc_p = pad_to(cond, n), pad_to(unc, n)
        x = lat[:, :, 0].permute(0, 2, 3, 1).contiguous().to(device)
        model.trim_context = False

        def enc():
            encode_prompt(pipe, args.prompt, device, dtype)
            encode_prompt(pipe, args.negative, device, dtype)

        def den():
            return model.denoise(cond_p, unc_p, x, args.steps, scale=4.0, shift=shift)

        out = den()

        def dec():
            return decode_ours(pipe, out)

        def ours_all():
            enc()
            o = den()
            decode_ours(pipe, o)

        s_hf = bench(hf, 1, args.e2e_iters, device)
        add_row(
            "end-to-end",
            f"diffusers pipeline ({args.steps} steps)",
            size,
            s_hf,
            s_hf["median"],
            None,
            peak_gb(hf, device),
        )
        s_all = bench(ours_all, 1, args.e2e_iters, device)
        add_row(
            "end-to-end",
            f"ours total ({args.steps} steps)",
            size,
            s_all,
            s_hf["median"],
            None,
            peak_gb(ours_all, device),
        )
        if args.fp8:
            model.enable_fp8()
            s_f = bench(ours_all, 1, args.e2e_iters, device)
            add_row(
                "end-to-end",
                f"ours total fp8 ({args.steps} steps)",
                size,
                s_f,
                s_hf["median"],
                None,
                peak_gb(ours_all, device),
            )
            model.disable_fp8()
        for name, fn in (
            ("ours: text encode", enc),
            ("ours: denoise", den),
            ("ours: decode", dec),
        ):
            st = bench(fn, 1, args.e2e_iters, device)
            add_row(
                "end-to-end",
                name,
                size,
                st,
                None,
                None,
                peak_gb(fn, device),
                f"{100 * st['median'] / s_all['median']:.0f}% of ours total",
            )


def run_train(model, args, dtype, device, size, profile_it):
    cfg = model.cfg
    H = W = size // 8
    lat = torch.randn(1, H, W, cfg.in_ch, device=device)
    ctx = torch.randn(1, args.ctx_len, cfg.text_dim, device=device, dtype=dtype)
    unc = torch.zeros(1, args.ctx_len, cfg.text_dim, device=device, dtype=dtype)
    params = add_lora(model, rank=args.lora_rank)
    flops = 3 * forward_flops(
        cfg, size, 1, args.ctx_len
    )  # forward + backward ~ 3x forward
    try:
        for ckpt in (True, False):
            model.checkpointing = ckpt

            def step():
                flow_loss(model, lat, ctx, unc, caption_dropout=0.0).backward()
                for p in params:
                    p.grad = None

            name = f"LoRA r{args.lora_rank} fwd+bwd, checkpointing {'on' if ckpt else 'off'}"
            try:
                stats = bench(step, 1, max(3, args.iters // 4), device)
                add_row(
                    "training",
                    name,
                    size,
                    stats,
                    None,
                    flops * (4 / 3 if ckpt else 1),
                    peak_gb(step, device),
                )
                if profile_it and not ckpt:
                    profile_call(
                        step,
                        f"training step {size} ({name})",
                        args.out,
                        device,
                        args.profile_iters,
                        args.top,
                        stats["median"],
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                add_row("training", name, size, None, note="OOM")
    finally:
        model.checkpointing = True
        remove_lora(model, merge=False)


def markdown(rows: list[dict]) -> str:
    out = []
    for sec in dict.fromkeys(r["section"] for r in rows):
        out += [
            f"\n### {sec}\n",
            "| size | implementation | ms | speedup vs diffusers | TFLOP/s | peak extra GiB | note |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in (r for r in rows if r["section"] == sec):
            f = lambda v, p=1: (
                "-"
                if v is None or (isinstance(v, float) and np.isnan(v))
                else f"{v:.{p}f}"
            )
            sp = "-" if r["speedup"] is None else f"{r['speedup']:.2f}x"
            out.append(
                f"| {r['size']} | {r['name']} | {f(r['ms'], 2)} | {sp} | {f(r['tflops'], 0)} | "
                f"{f(r['mem_gb'], 2)} | {r['note']} |"
            )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Benchmark and profile the converted model against diffusers.

    Measures transformer forward latency, end-to-end generation time, peak GPU memory, and
    (optionally) LoRA training steps, then profiles the forward pass of both implementations
    and prints where GPU time goes. Results are saved as a Markdown report, JSON, and Chrome traces.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on completion.
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--repo", default="circlestone-labs/Anima-Base-v1.0-Diffusers")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument(
        "--batch", type=int, default=2, help="Forward batch (2 = one CFG step)."
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--e2e-iters", type=int, default=3)
    ap.add_argument("--profile-iters", type=int, default=3)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument(
        "--peak-tflops", type=float, default=989.0, help="Dense bf16 peak of the GPU."
    )
    ap.add_argument(
        "--prompt", default="masterpiece, best quality, 1girl, solo, city lights"
    )
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--qk-eps", type=float, default=1e-5)
    ap.add_argument(
        "--hf-compile",
        action="store_true",
        help="Also time diffusers under torch.compile.",
    )
    ap.add_argument(
        "--train", action="store_true", help="Also time LoRA training steps."
    )
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument(
        "--fp8",
        action="store_true",
        help="Also measure the FP8 path in the forward and end-to-end sections.",
    )
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument(
        "--profile-size",
        type=int,
        default=None,
        help="Image size that gets the kernel breakdown; defaults to the first of --sizes.",
    )
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--out", default="profile_out")
    args = ap.parse_args(argv)

    dtype, device = getattr(torch, args.dtype), args.device
    os.makedirs(args.out, exist_ok=True)
    pipe = ModularPipeline.from_pretrained(args.repo)
    pipe.load_components(dtype=dtype)
    pipe.to(device)
    tr = pipe.transformer.eval()
    model = convert_copy(tr, dtype, device, args.qk_eps)
    pipe.vae.fast = VAE(pipe.vae).to(device, dtype).eval()

    gpu = torch.cuda.get_device_name() if device.startswith("cuda") else "cpu"
    env = {
        "gpu": gpu,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "fa4": bool(use_fa4(torch.empty(1, device=device, dtype=dtype))),
        "quack": have_quack() if device.startswith("cuda") else False,
        "dtype": args.dtype,
        "weights_gb": {"diffusers": weights_gb(tr), "ours": weights_gb(model)},
    }
    print("## Environment\n")
    for k, v in env.items():
        print(f"- {k}: {v}")

    hf_fwd = torch.compile(tr) if args.hf_compile else tr
    for size in args.sizes:
        print(f"\n## {size}x{size}")
        flops = forward_flops(model.cfg, size, args.batch, args.ctx_len)
        run_forward(pipe, tr, model, args, dtype, device, size, flops)
        if args.hf_compile:
            x, t, ctx, pad = make_inputs(
                model.cfg, size, args.batch, args.ctx_len, dtype, device
            )
            with torch.no_grad():
                s = bench(
                    lambda: hf_fwd(
                        hidden_states=x,  # noqa
                        timestep=t,  # noqa
                        encoder_hidden_states=ctx,  # noqa
                        padding_mask=pad,  # noqa
                        return_dict=False,
                    ),
                    args.warmup,
                    args.iters,
                    device,
                )
            base = next(
                r["ms"] for r in ROWS if r["name"] == "diffusers" and r["size"] == size
            )
            add_row("forward", "diffusers (torch.compile)", size, s, base, flops)
        if not args.skip_e2e:
            run_e2e(pipe, model, args, dtype, device, size)
        if args.train:
            run_train(
                model,
                args,
                dtype,
                device,
                size,
                profiled(args, size),
            )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    report = markdown(ROWS)
    print("\n" + report)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(f"```\n{json.dumps(env, indent=2)}\n```\n{report}\n")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"env": env, "rows": ROWS}, f, indent=2)
    print(f"\nSaved report, JSON and Chrome traces to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
