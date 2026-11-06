import os

import torch
import torch.distributed as dist


def setup() -> tuple[int, int, torch.device]:
    """Initialize multi-GPU training when launched by `torchrun`, otherwise use one GPU.

    Returns:
        `(rank, world_size, device)` for this process.
    """
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if "RANK" not in os.environ:
        return 0, 1, device
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), device


def broadcast_params(params: list[torch.nn.Parameter]) -> None:
    """Copy rank 0's values to every rank.

    Adapters are initialized randomly per process, so call this once after `add_lora`.

    Args:
        params: Parameters, in the same order on every rank.
    """
    with torch.no_grad():
        for p in params:
            dist.broadcast(p, 0)


def average_grads(params: list[torch.nn.Parameter]) -> None:
    """Average gradients across ranks, in place.

    Call after the last micro-batch's `backward()` and before clipping. A parameter without a
    gradient on this rank counts as zero.

    Args:
        params: Parameters, in the same order on every rank.
    """
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    flat = torch.cat([p.grad.reshape(-1) for p in params])  # type: ignore
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    offset = 0
    for p in params:
        p.grad.copy_(flat[offset : offset + p.numel()].view_as(p))  # type: ignore
        offset += p.numel()
