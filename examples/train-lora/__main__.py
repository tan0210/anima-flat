import argparse

import torch
import torch.distributed as dist

from src.lora import add_lora, save_lora
from src.model import load
from src.train import flow_loss

from .data import load_dataset
from .distributed import average_grads, broadcast_params, setup
from .schedule import lr_at


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2, help="per GPU")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--out", default="adapter.safetensors")
    args = ap.parse_args()

    rank, world, device = setup()
    distributed = world > 1

    model = load(args.checkpoint, device=device)
    model.checkpointing = True
    params = add_lora(model, rank=args.rank)
    if distributed:
        broadcast_params(params)
    opt = torch.optim.AdamW(params, lr=args.lr)

    data = load_dataset(args.data)
    uncond = data.uncond.to(device)
    # Each rank draws its own batches, noise levels and noise.
    torch.manual_seed(1234 + rank)

    model.train()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        total = torch.zeros((), device=device)
        for _ in range(args.accum):
            latents, context = data.batch(args.batch, device)
            loss = flow_loss(model, latents, context, uncond)
            (loss / args.accum).backward()
            total += loss.detach() / args.accum
        if distributed:
            average_grads(params)
            dist.all_reduce(total, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if rank == 0 and step % 10 == 0:
            print(f"step {step} loss {total.item():.4f}")

    if rank == 0:
        save_lora(model, args.out)
    if distributed:
        dist.destroy_process_group()


main()
