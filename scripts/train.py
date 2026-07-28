# import logging
from loguru import logger
import sys
import time
import warnings
import os
import torch.multiprocessing as mp
import torch.distributed as dist


import trainers
from trainers.utils import config
from trainers.utils import tools as other_tools
from trainers.utils.wandb_logger import WandbLogger

from trainers.utils.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)

# logger = logging.getLogger()

# @logger.catch
def main_worker(args):
    other_tools.set_random_seed(args)
    
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # Initialize the process group using environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if "LOCAL_RANK" in os.environ:
        # torchrun-launched (multi-GPU DDP)
        set_device()
        logger.info("Going to init comms...")
        dist.init_process_group(backend=BACKEND)
    else:
        # Plain `python scripts/train.py` (single GPU, no DDP)
        logger.info("LOCAL_RANK not set: single-process mode, skipping dist.init_process_group.")
        set_device()

    world_size = get_world_size()
    global_rank = get_rank()

    if world_size > 1:
        args.ddp = True  # Ensure ddp flag is set
    else:
        args.ddp = False

    logger.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not Set')}")
    
    other_tools.set_args_and_logger(args, global_rank)
    if global_rank == 0:
        other_tools.print_exp_info(args)
    
    # return one intance of trainer
    logger.info(f"Rank {global_rank} trainer initializing...")
    trainer = getattr(trainers, args.trainer+"Trainer")(args)
    trainer.wandb_logger = WandbLogger.from_trainer(
        args,
        trainer,
        rank=global_rank,
        world_size=world_size,
        job_type="train",
    )
    if args.is_continue:
        # logger.info("Continue training ...")
        # other_tools.load_checkpoints(trainer.model, args.continue_ckpt, after_distributed=True)
        start_epoch = int(os.path.basename(args.continue_ckpt).split("_")[1].replace(".safetensors", "")) + 1
        if global_rank == 0: logger.info(f"Start from epoch {start_epoch}")
    else:
        start_epoch = 0
        if global_rank == 0: logger.info("Training from scratch ...")
    start_time = time.time()
    try:
        for epoch in range(start_epoch, args.epochs+1):
            epoch_start_time = time.time()
            if args.ddp:
                trainer.val_loader.sampler.set_epoch(epoch)

            if epoch != args.epochs:
                if args.ddp:
                    trainer.train_loader.sampler.set_epoch(epoch)
                trainer.tracker.reset()
                trainer.train(epoch)
            if args.debug:
                # Validation contains DDP collectives, so every rank must enter.
                trainer.val(epoch)
                if global_rank == 0:
                    debug_checkpoint = os.path.join(trainer.checkpoint_path, f"last_{epoch}.safetensors")
                    other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}"), trainer.model, opt=None, epoch=None, lrs=None, save_dtype=args.param_dtype)
                    other_tools.load_checkpoints(trainer.model, debug_checkpoint, after_distributed=True)
                    trainer.wandb_logger.log_checkpoint(epoch=epoch, path=debug_checkpoint)
                # trainer.test(epoch)

            if epoch % args.test_period == 0:
                if global_rank == 0:
                    logger.info(f"[GPU {global_rank}] Saving checkpoints:")
                    checkpoint_path = os.path.join(trainer.checkpoint_path, f"last_{epoch}.safetensors")
                    other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}"), trainer.model, opt=None, epoch=None, lrs=None, save_dtype=args.param_dtype)
                    # trainer.test(epoch)
                    args.test_ckpt = checkpoint_path
                    other_tools.update_args_file(args, rank=global_rank)
                    trainer.wandb_logger.log_checkpoint(epoch=epoch, path=checkpoint_path)

            if epoch % args.test_period == 0 and epoch != 0 and not args.debug:
                logger.info(f"GPU {global_rank} Validation:")
                trainer.val(epoch)

            epoch_seconds = time.time() - epoch_start_time
            elapsed_seconds = time.time() - start_time
            completed_epochs = max(1, epoch - start_epoch + 1)
            remaining_epochs = max(0, args.epochs - epoch)
            remaining_seconds = (
                elapsed_seconds / completed_epochs * remaining_epochs
            )
            if trainer.global_rank == 0:
                logger.info(
                    f"Time info >>>>  elapsed: {elapsed_seconds/60:.2f} mins\t"
                    f"remain: {remaining_seconds/60:.2f} mins"
                )
                trainer.wandb_logger.log_epoch(
                    epoch=epoch,
                    epoch_seconds=epoch_seconds,
                    elapsed_seconds=elapsed_seconds,
                    remaining_seconds=remaining_seconds,
                )
    finally:
        trainer.wandb_logger.finish()
        writer = getattr(trainer, "writer", None)
        if writer is not None:
            writer.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    

    
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    args = config.parse_args()
    args.is_train = True
    main_worker(args)
