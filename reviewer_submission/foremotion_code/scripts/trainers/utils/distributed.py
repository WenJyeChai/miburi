from loguru import logger
import os
from functools import lru_cache
from typing import List, Union

import torch
import torch.distributed as dist

# logger = logging.getLogger("distributed")

BACKEND = "nccl"


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


@lru_cache()
def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


@lru_cache()
def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def visible_devices() -> List[int]:
    return [int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]


def set_device():
    """Bind the calling process to a CUDA device.

    Works in two modes:
    - torchrun-launched: LOCAL_RANK is set; binds to that rank's GPU.
    - single-process (`python scripts/train.py`): binds to cuda:0.
    """
    if "LOCAL_RANK" in os.environ:
        # torchrun path
        logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        logger.info(
            f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
        logger.info(f"local rank: {int(os.environ['LOCAL_RANK'])}")
        logger.info(f"global rank: {int(os.environ['RANK'])}")

        assert torch.cuda.is_available()

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            assert len(visible_devices()) == torch.cuda.device_count()

        if torch.cuda.device_count() == 1:
            torch.cuda.set_device(0)
            return

        local_rank = int(os.environ["LOCAL_RANK"])
        logger.info(f"Set cuda device to {local_rank}")
        assert 0 <= local_rank < torch.cuda.device_count(), (
            local_rank,
            torch.cuda.device_count(),
        )
        torch.cuda.set_device(local_rank)
        return

    # Single-process path: plain `python scripts/train.py`
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info(
            "Single-process mode: bound to cuda:0 "
            f"(torch.cuda.device_count={torch.cuda.device_count()})"
        )
    else:
        logger.warning("Single-process mode: no CUDA available; running on CPU.")


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    if not is_distributed():
        return float(metric)
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def reduce_sum_count(
    value_sum: Union[float, int],
    value_count: Union[float, int],
    *,
    device: Union[int, torch.device, str, None] = None,
) -> tuple[float, int]:
    """Sum an accumulator and its count across all distributed ranks."""
    if not is_distributed():
        return float(value_sum), int(value_count)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    stats = torch.tensor(
        [float(value_sum), float(value_count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item())


def sync_tracker_meters(
    tracker,
    *,
    states: tuple[str, ...] = ("train", "val"),
    device: Union[int, torch.device, str, None] = None,
) -> None:
    """Make selected EpochTracker meters represent all DDP ranks.

    Every rank executes collectives in the same metric/state order, including
    meters whose local count is zero.
    """
    if not is_distributed():
        return
    for name in tracker.metric_names:
        for state in states:
            meter = tracker.loss_meters[name][state]
            meter.sum, meter.count = reduce_sum_count(
                meter.sum,
                meter.count,
                device=device,
            )
            meter.avg = meter.sum / meter.count if meter.count else 0.0


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ
