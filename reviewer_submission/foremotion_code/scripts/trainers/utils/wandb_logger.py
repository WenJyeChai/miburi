"""Optional Weights & Biases logging for MIBURI trainers.

Only rank 0 owns a W&B run. This keeps DDP training to a single W&B client
while the existing TensorBoard logging remains available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from loguru import logger


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    return str(value)


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


class WandbLogger:
    """Small rank-aware wrapper around a W&B run."""

    def __init__(
        self,
        args,
        *,
        rank: int,
        world_size: int,
        checkpoint_path: str,
        model=None,
        train_samples: int | None = None,
        val_samples: int | None = None,
        test_samples: int | None = None,
        job_type: str | None = None,
    ) -> None:
        self.run = None
        self._warned_log_failure = False

        if not bool(getattr(args, "wandb", False)):
            return

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging was enabled, but the 'wandb' package is not installed. "
                "Install it with: pip install -e '.[tracking]'"
            ) from exc

        # All ranks verify the optional dependency above; only rank 0 opens a
        # client connection and owns the actual run.
        if rank != 0:
            return

        checkpoint_dir = Path(checkpoint_path).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        run_name = getattr(args, "wandb_name", None)
        if not run_name:
            run_name = f"{Path(str(args.name)).name}{getattr(args, 'notes', '')}"

        init_kwargs: dict[str, Any] = {
            "project": getattr(args, "wandb_project", "miburi"),
            "entity": getattr(args, "wandb_entity", None),
            "name": run_name,
            "group": getattr(args, "wandb_group", None),
            "job_type": job_type or getattr(args, "wandb_job_type", "train"),
            "tags": getattr(args, "wandb_tags", None) or None,
            "mode": getattr(args, "wandb_mode", "online"),
            "dir": str(checkpoint_dir),
            "config": {
                key: _serializable(value)
                for key, value in vars(args).items()
            },
        }

        run_id = getattr(args, "wandb_run_id", None)
        resume = getattr(args, "wandb_resume", "never")
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = resume
        elif resume == "auto":
            init_kwargs["resume"] = "auto"

        self.run = wandb.init(**init_kwargs)
        self.run.define_metric("trainer/global_step")
        self.run.define_metric("trainer/epoch")
        self.run.define_metric("train/*", step_metric="trainer/global_step")
        self.run.define_metric("performance/*", step_metric="trainer/global_step")
        self.run.define_metric("system/*", step_metric="trainer/global_step")
        self.run.define_metric("val/*", step_metric="trainer/epoch")
        self.run.define_metric("epoch_train/*", step_metric="trainer/epoch")
        self.run.define_metric("best/*", step_metric="trainer/epoch")
        self.run.define_metric("epoch/*", step_metric="trainer/epoch")
        self.run.define_metric("eval/*")

        parameter_count = None
        trainable_parameter_count = None
        if model is not None:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            trainable_parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )

        summary_values = {
            "world_size": world_size,
            "batch_size_per_gpu": int(getattr(args, "batch_size", 1)),
            "global_batch_size": int(getattr(args, "batch_size", 1)) * world_size,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "parameters": parameter_count,
            "trainable_parameters": trainable_parameter_count,
            "checkpoint_directory": str(checkpoint_dir),
        }
        for key, value in summary_values.items():
            if value is not None:
                self.run.summary[key] = value

        logger.info(
            "W&B initialized: project={} run={} mode={}",
            getattr(args, "wandb_project", "miburi"),
            self.run.name,
            getattr(args, "wandb_mode", "online"),
        )

    @classmethod
    def from_trainer(
        cls,
        args,
        trainer,
        *,
        rank: int,
        world_size: int,
        job_type: str | None = None,
    ) -> "WandbLogger":
        return cls(
            args,
            rank=rank,
            world_size=world_size,
            checkpoint_path=trainer.checkpoint_path,
            model=getattr(trainer, "model", None),
            train_samples=(
                len(trainer.train_data)
                if hasattr(trainer, "train_data")
                else None
            ),
            val_samples=(
                len(trainer.val_data)
                if hasattr(trainer, "val_data")
                else None
            ),
            test_samples=(
                len(trainer.test_data)
                if hasattr(trainer, "test_data")
                else None
            ),
            job_type=job_type,
        )

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def _log(self, values: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.log(dict(values))
        except Exception as exc:
            if not self._warned_log_failure:
                logger.warning("W&B metric logging failed: {}", exc)
                self._warned_log_failure = True

    def log_train(
        self,
        *,
        tracker,
        epoch: int,
        iteration: int,
        global_step: int,
        learning_rate: float,
        data_time_seconds: float,
        train_time_seconds: float,
        memory_gb: float,
        global_batch_size: int,
        discriminator_learning_rate: float | None = None,
    ) -> None:
        if self.run is None:
            return

        values: dict[str, Any] = {
            "trainer/global_step": global_step,
            "trainer/epoch": epoch,
            "trainer/iteration": iteration,
            "train/learning_rate": learning_rate,
            "performance/data_time_ms": data_time_seconds * 1000.0,
            "performance/train_time_ms": train_time_seconds * 1000.0,
            "system/gpu_memory_reserved_gb": memory_gb,
        }
        total_time = data_time_seconds + train_time_seconds
        if total_time > 0:
            values["performance/samples_per_second"] = global_batch_size / total_time
        if discriminator_learning_rate is not None:
            values["train/discriminator_learning_rate"] = discriminator_learning_rate

        for name, states in tracker.loss_meters.items():
            meter = states["train"]
            if meter.count > 0:
                values[f"train/{name}"] = _as_float(meter.avg)
        self._log(values)

    def log_validation(
        self,
        *,
        tracker,
        epoch: int,
        extra_values: Mapping[str, Any] | None = None,
    ) -> None:
        if self.run is None:
            return

        values: dict[str, Any] = {"trainer/epoch": epoch}
        for name, states in tracker.loss_meters.items():
            train_meter = states["train"]
            val_meter = states["val"]
            if train_meter.count > 0:
                values[f"epoch_train/{name}"] = _as_float(train_meter.avg)
            if val_meter.count > 0:
                values[f"val/{name}"] = _as_float(val_meter.avg)
                best = tracker.values[name]["val"]["best"]
                values[f"best/{name}"] = _as_float(best["value"])
                values[f"best/{name}_epoch"] = int(best["epoch"])
        if extra_values:
            for name, value in extra_values.items():
                values[f"val/{name}"] = _as_float(value)
        self._log(values)

    def log_epoch(
        self,
        *,
        epoch: int,
        epoch_seconds: float,
        elapsed_seconds: float,
        remaining_seconds: float,
    ) -> None:
        self._log(
            {
                "trainer/epoch": epoch,
                "epoch/duration_seconds": epoch_seconds,
                "epoch/elapsed_seconds": elapsed_seconds,
                "epoch/remaining_seconds": remaining_seconds,
            }
        )

    def log_checkpoint(self, *, epoch: int, path: str) -> None:
        if self.run is None:
            return
        self.run.summary["latest_checkpoint_epoch"] = epoch
        self.run.summary["latest_checkpoint_path"] = str(Path(path).resolve())

    def log_evaluation(
        self,
        metrics: Mapping[str, Any] | None,
        *,
        checkpoint: str | None = None,
    ) -> None:
        if self.run is None or not metrics:
            return
        values = {
            f"eval/{name}": _as_float(value)
            for name, value in metrics.items()
        }
        self._log(values)
        if checkpoint:
            self.run.summary["evaluated_checkpoint"] = str(Path(checkpoint).resolve())

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
