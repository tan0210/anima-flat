import argparse
import json
import math
import os
import re
from collections.abc import Sequence

import matplotlib
import numpy as np
import torch
from PIL import Image

from ..sample import noise
from .generate import (
    add_args,
    check_size,
    decode_latents,
    denoise,
    embed,
    load_models,
    to_uint8,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.dpi": 100,
    "savefig.facecolor": "white",
}
VALUE_SIZE = 9.5
COLORS = ["#3b75af", "#ef8636", "#519e3e", "#c03d3e", "#8d6ab8"]
FIGSIZE = (7, 4.2)

VARIANTS = ("diffusers", "flat", "flat-fp8")
VARIANT_LABELS = {
    "diffusers": "Diffusers (stock pipeline)",
    "flat": "Flat (bf16)",
    "flat-fp8": "Flat (FP8)",
}

FORWARD_NAMES = {
    "diffusers": "Diffusers",
    "diffusers (torch.compile)": "Diffusers (compiled)",
    "ours fused": "Flat (bf16)",
    "ours fused (fp8)": "Flat (FP8)",
}
E2E_NAMES = {
    "diffusers pipeline": "Diffusers",
    "ours total": "Flat (bf16)",
    "ours total fp8": "Flat (FP8)",
}


def grouped_bars(
    groups: Sequence[str],
    series: dict[str, list[float | None]],
    xlabel: str,
    ylabel: str,
    title: str,
    ax=None,
    fmt: str = "{:.1f}",
    missing: str = "",
):
    """Draw grouped bars with the value written above each bar.

    Args:
        groups: One label per group on the x-axis.
        series: Bars of each group, by legend label. `None` leaves a gap, annotated with `missing`.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Axes title.
        ax: Axes to draw on; a new figure is created if omitted.
        fmt: Format of the value labels.
        missing: Text shown where a value is `None`, for example `"OOM"`.

    Returns:
        The figure when `ax` is omitted, otherwise None.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    top = max((v for vs in series.values() for v in vs if v is not None), default=1.0)
    for i, ((label, values), color) in enumerate(
        zip(series.items(), COLORS, strict=False)
    ):
        pos = x + (i - (n - 1) / 2) * width
        for p, v in zip(pos, values, strict=True):
            if v is None:
                if missing:
                    ax.annotate(
                        missing,
                        (p, 0),
                        ha="center",
                        va="bottom",
                        fontsize=VALUE_SIZE,
                        xytext=(0, 2),
                        textcoords="offset points",
                    )
                continue
            ax.bar(p, v, width * 0.92, color=color, label=label if label else None)
            ax.annotate(
                fmt.format(v),
                (p, v),
                ha="center",
                va="bottom",
                fontsize=VALUE_SIZE,
                xytext=(0, 2),
                textcoords="offset points",
            )
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles, strict=True))
    cols = min(len(seen), 2)
    ax.legend(seen.values(), seen.keys(), loc="upper left", ncol=cols)
    ax.set_xticks(x, list(groups))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    rows = math.ceil(len(seen) / cols)
    ax.set_ylim(0, top * (1.2 + 0.13 * rows))
    return fig


def rows_of(data: dict, section: str) -> list[dict]:
    """Select the rows of one section of a `profile` results file.

    Args:
        data: Parsed `results.json`.
        section: Section name, such as `"forward"`.

    Returns:
        The rows of that section.
    """
    return [r for r in data["rows"] if r["section"] == section]


def by_size(
    rows: list[dict], names: dict[str, str]
) -> tuple[list[int], dict[str, list[float | None]]]:
    """Arrange rows as one series per implementation and one group per image size.

    Args:
        rows: Rows from `rows_of`.
        names: Row name to legend label; rows with other names are ignored.

    Returns:
        The sizes in ascending order and the latency in milliseconds of each series at each size.
    """
    sizes = sorted({r["size"] for r in rows})
    out: dict[str, list[float | None]] = {}
    for name, label in names.items():
        found = {r["size"]: r["ms"] for r in rows if r["name"] == name}
        if found:
            out[label] = [found.get(s) for s in sizes]
    return sizes, out


def e2e_series(rows: list[dict]) -> tuple[list[int], dict, int | None]:
    """Arrange end-to-end rows, whose names carry the step count.

    Args:
        rows: End-to-end rows.

    Returns:
        Sizes, series, and the number of sampling steps (None if the rows do not say).
    """
    steps = None
    renamed = []
    for r in rows:
        m = re.fullmatch(r"(.*?) \((\d+) steps\)", r["name"])
        if m and m.group(1) in E2E_NAMES:
            steps = int(m.group(2))
            renamed.append({**r, "name": m.group(1)})
    sizes, series = by_size(renamed, E2E_NAMES)
    return sizes, series, steps


def speed_figures(args: argparse.Namespace) -> list[str]:
    """Draw the benchmark figures from measured results.

    Args:
        args: Options with `profile`, `throughput`, `out`, `dpi`.

    Returns:
        Paths of the figures written.
    """
    data: dict = {"rows": []}
    for path in args.profile:
        with open(path, encoding="utf-8") as f:
            part = json.load(f)
        have = {r["section"] for r in data["rows"]}
        data["rows"] += [r for r in part["rows"] if r["section"] not in have]
    os.makedirs(args.out, exist_ok=True)
    written = []

    def save(fig, name: str) -> None:
        path = os.path.join(args.out, name)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    sizes, series = by_size(rows_of(data, "forward"), FORWARD_NAMES)
    if series:
        save(
            grouped_bars(
                [str(s) for s in sizes],
                series,
                "Resolution (px)",
                "Forward latency (ms)",
                "Forward pass",
            ),
            "forward.png",
        )

    sizes, series, steps = e2e_series(rows_of(data, "end-to-end"))
    if series:
        extra = f", {steps} steps" if steps else ""
        save(
            grouped_bars(
                [str(s) for s in sizes],
                series,
                "Resolution (px)",
                "Latency (ms)",
                f"End-to-end generation{extra}",
                fmt="{:,.0f}",
            ),
            "end_to_end.png",
        )

    train = rows_of(data, "training")
    if train:
        sizes = sorted({r["size"] for r in train})
        series = {}
        for tag, label in (("on", "Checkpointing on"), ("off", "Checkpointing off")):
            found = {
                r["size"]: r["ms"]
                for r in train
                if r["name"].endswith(f"checkpointing {tag}")
            }
            if found:
                series[label] = [found.get(s) for s in sizes]
        save(
            grouped_bars(
                [str(s) for s in sizes],
                series,
                "Resolution (px)",
                "Training step (ms)",
                "LoRA fine-tuning step",
                missing="OOM",
            ),
            "training.png",
        )

    if args.throughput:
        runs = []
        for path in args.throughput:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        batches = sorted({r["batch"] for run in runs for r in run["rows"]})
        per_s: dict[str, list[float | None]] = {}
        cost: dict[str, list[float | None]] = {}
        for run in runs:
            label = "Flat (FP8)" if run.get("fp8") else "Flat (bf16)"
            found = {r["batch"]: r for r in run["rows"]}
            per_s[label] = [
                found[b]["images_per_s"] if b in found else None for b in batches
            ]
            cost[label] = [
                found[b]["usd_per_1000"] if b in found else None for b in batches
            ]
        s = runs[0]["settings"]
        hf = next(
            (
                r["diffusers_ms_per_image"]
                for r in runs
                if r.get("diffusers_ms_per_image")
            ),
            None,
        )
        if hf:
            gap = [hf if b == batches[0] else None for b in batches]
            per_s = {"Diffusers": [1e3 / v if v else None for v in gap], **per_s}
            cost = {
                "Diffusers": [s["price"] / 3600 * v if v else None for v in gap],
                **cost,
            }
        fig, axs = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 1.7, FIGSIZE[1]))
        grouped_bars(
            [str(b) for b in batches],
            per_s,
            "Batch size",
            "Images per second",
            "Throughput",
            ax=axs[0],
            fmt="{:.2f}",
        )
        grouped_bars(
            [str(b) for b in batches],
            cost,
            "Batch size",
            f"USD per 1,000 images (${s['price']}/h)",
            "Cost",
            ax=axs[1],
            fmt="${:.2f}",
        )
        fig.suptitle(
            f"{s['width']}×{s['height']}, {s['steps']} steps, CFG {s['cfg']}",
            fontsize=VALUE_SIZE + 2.5,
        )
        fig.tight_layout()
        save(fig, "throughput.png")
    return written


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio between two 8-bit images.

    Args:
        a: Image array of `uint8`.
        b: Image array of the same shape.

    Returns:
        PSNR in dB; infinity for identical images.
    """
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


@torch.no_grad()
def diffusers_image(
    pipe, args: argparse.Namespace, seed: int, channels: int
) -> np.ndarray:
    """Generate one image with the stock pipeline from the same starting noise as the flat model.

    Args:
        pipe: The loaded pipeline.
        args: Generation options.
        seed: Noise seed.
        channels: Latent channels.

    Returns:
        `uint8` array `[H, W, 3]`. The pipeline uses its own scheduler and shift, not `--sampler`
        and `--shift`.
    """
    if getattr(pipe, "guider", None) is not None:
        pipe.guider.guidance_scale = args.cfg
    x = noise(1, args.height // 8, args.width // 8, channels, seed, args.device)
    lat = x.permute(0, 3, 1, 2)[:, :, None].contiguous()
    img = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        latents=lat,
    ).images[0]
    return np.asarray(img.convert("RGB"))


@torch.no_grad()
def flat_image(
    pipe, model, cond, unc, args: argparse.Namespace, seed: int
) -> np.ndarray:
    """Generate one image with the flat model.

    Args:
        pipe: The loaded pipeline (for decoding).
        model: The flat model.
        cond: Positive embeddings.
        unc: Negative embeddings.
        args: Generation options.
        seed: Noise seed.

    Returns:
        `uint8` array `[H, W, 3]`.
    """
    x = denoise(model, cond, unc, args, [seed])
    return to_uint8(decode_latents(pipe, x))[0]


def image_pair(args: argparse.Namespace) -> list[str]:
    """Generate one image per variant from the same prompt and seed and save each as its own PNG.

    Args:
        args: Generation options plus `variants` and `out`.

    Returns:
        Paths of the images written.

    Raises:
        SystemExit: If a variant needs a Hopper GPU that is not present, or `--fp8` is passed.
    """
    check_size(args)
    if args.fp8:
        raise SystemExit(
            "--fp8 does not apply here; choose it with --variants flat-fp8"
        )
    dtype = getattr(torch, args.dtype)
    pipe, model = load_models(args)
    cond, unc = embed(pipe, args.prompt, args.negative, args.device, dtype)
    os.makedirs(args.out, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    paths = []
    for v in args.variants:
        if v == "diffusers":
            arr = diffusers_image(pipe, args, args.seed, model.cfg.in_ch)
        else:
            if v == "flat-fp8":
                model.enable_fp8()
            else:
                model.disable_fp8()
            arr = flat_image(pipe, model, cond, unc, args, args.seed)
            model.disable_fp8()
        arrays[v] = arr
        path = os.path.join(args.out, f"{v}.png")
        Image.fromarray(arr).save(path)
        paths.append(path)
    first = args.variants[0]
    print(
        f"seed {args.seed}, {args.width}x{args.height}, {args.steps} steps, {args.sampler}\n"
    )
    print("| image | PSNR vs " + VARIANT_LABELS[first] + " |")
    print("|---|---|")
    for v in args.variants:
        value = "-" if v == first else f"{psnr(arrays[v], arrays[first]):.2f} dB"
        print(f"| {VARIANT_LABELS[v]} | {value} |")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Make the figures used in the README.

    `speed` draws bar charts from the JSON written by `profile` and `throughput`. `images`
    generates the same prompt and seed with each chosen variant and saves every image as its own
    PNG, with no grid or captions.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on success.
    """
    plt.rcParams.update(STYLE)  # type: ignore
    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("speed", help="Bar charts from benchmark results.")
    sp.add_argument(
        "--profile",
        nargs="+",
        required=True,
        help="`results.json` files written by `profile`; a section found in an earlier "
        "file is not taken from a later one.",
    )
    sp.add_argument(
        "--throughput",
        nargs="+",
        default=[],
        help="JSON files written by `throughput --json`; one per precision.",
    )
    sp.add_argument("--out", default="figures")
    sp.add_argument("--dpi", type=int, default=200)

    ip = sub.add_parser("images", help="Same prompt and seed through each variant.")
    add_args(ip)
    ip.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=["flat", "flat-fp8"],
        help="Images to make; the first is the reference for PSNR.",
    )
    ip.add_argument("--out", default="compare_out")

    args = ap.parse_args(argv)
    paths = speed_figures(args) if args.command == "speed" else image_pair(args)
    print("\n" + "\n".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
