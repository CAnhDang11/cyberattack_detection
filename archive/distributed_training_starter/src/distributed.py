"""Distributed helpers for PyTorch Distributed Data Parallel (DDP)."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup_distributed():
    """Initialize DDP when launched with torchrun; otherwise run single-process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        # On macOS or Linux when using Gloo, force IPv4 interface to avoid IPv6 resolution hangs
        if backend == "gloo" and "GLOO_SOCKET_IFNAME" not in os.environ:
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            if master_addr in ("127.0.0.1", "localhost", ""):
                os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
            else:
                os.environ["GLOO_SOCKET_IFNAME"] = "en0"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=30),
        )

    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def wrap_ddp(model, device, distributed: bool, local_rank: int):
    if not distributed:
        return model
    if device.type == "cuda":
        return DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    return DistributedDataParallel(model)


def is_main_process(rank: int) -> bool:
    return rank == 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_sum(tensor, distributed: bool):
    """Sum a tensor across ranks in-place and return it."""
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def barrier(distributed: bool):
    if distributed and dist.is_initialized():
        dist.barrier()
