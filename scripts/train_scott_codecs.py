"""Launch independent Scott-only codec training runs with fresh initialization.

Run from the repository root, for example::

    python scripts/train_scott_codecs.py --part all --wandb True

The launcher runs the existing codec trainers sequentially. It never starts from
the released weights, and it does not train the gesture language model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
PARTS = ("upper", "lower", "face")
CONFIGS = {part: ROOT / f"configs/mimi_scott_scratch_{part}.yaml" for part in PARTS}


def _boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=(*PARTS, "all"), default="all",
                        help="Train one codec, or all three sequentially on the same GPU.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without starting training.")
    for name in ("batch_size", "loader_workers", "epochs", "test_period", "log_period", "random_seed"):
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.add_argument("--lr_base", type=float, default=None)
    for name in ("beatx_cache_path", "deps_path", "out_path"):
        parser.add_argument(f"--{name}", default=None)
    parser.add_argument("--codec_eval_fgd", type=_boolean, default=False)
    parser.add_argument("--wandb", type=_boolean, default=False)
    for name in ("wandb_project", "wandb_entity", "wandb_group", "wandb_name"):
        parser.add_argument(f"--{name}", default=None)
    parser.add_argument("--wandb_tags", nargs="+", default=None)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default=None)
    args = parser.parse_args(argv)
    for name in ("batch_size", "epochs", "test_period", "log_period"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name} must be positive.")
    if args.loader_workers is not None and args.loader_workers < 0:
        parser.error("--loader_workers cannot be negative.")
    if args.lr_base is not None and (not math.isfinite(args.lr_base) or args.lr_base <= 0):
        parser.error("--lr_base must be finite and positive.")
    return args


def _path(value, directory=False):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    result = path.resolve().as_posix()
    return result.rstrip("/") + "/" if directory else result


def build_commands(args, run_id=None):
    """Return argv lists; a supplied run_id makes command construction testable."""
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run_id must contain only letters, digits, underscores and hyphens.")
    parts = PARTS if args.part == "all" else (args.part,)
    commands = []
    for part in parts:
        command = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(CONFIGS[part])]
        for name in ("batch_size", "loader_workers", "epochs", "test_period", "log_period",
                     "random_seed", "lr_base", "wandb_project", "wandb_entity", "wandb_group", "wandb_mode"):
            value = getattr(args, name)
            if value is not None:
                command.extend([f"--{name}", str(value)])
        for name in ("beatx_cache_path", "deps_path", "out_path"):
            value = getattr(args, name)
            if value is not None:
                resolved = _path(value, directory=name != "beatx_cache_path")
                if name == "beatx_cache_path" and Path(resolved).is_dir():
                    resolved = str(Path(resolved) / "database.hdf5")
                command.extend([f"--{name}", resolved])
        # Keep each part and invocation separate even when launched in the same minute.
        run_name = args.wandb_name or "codec-scott-scratch"
        command.extend([
            "--notes", f"_{run_id}", "--is_train", "True", "--is_continue", "False",
            "--dataset_ratio", "scott_beatx_lowervalid", "--codec_standard_eval", "True",
            "--codec_eval_fgd", str(args.codec_eval_fgd), "--wandb", str(args.wandb),
            "--wandb_name", f"{run_name}-{part}-{run_id}", "--wandb_resume", "never",
        ])
        if args.wandb_tags is not None:
            tags = list(dict.fromkeys(["codec", "scott", "scratch", part, *args.wandb_tags]))
            command.extend(["--wandb_tags", *tags])
        commands.append(command)
    return commands


def preflight(args):
    """Check local inputs before starting any part, without opening the test set."""
    cache = Path(_path(args.beatx_cache_path or "datasets/data_cache/beatx_gtdm3/database.hdf5"))
    if cache.is_dir():
        cache = cache / "database.hdf5"
    deps = Path(_path(args.deps_path or "assets_dep"))
    body_model = deps / "smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz"
    missing = [str(path) for path in (cache, body_model) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required training input is missing:\n  " + "\n  ".join(missing))
    if "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Run this sequential single-GPU launcher with python, not torchrun.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The existing codec trainers require CUDA. Use --dry_run to inspect commands.")


def main(argv=None):
    args = parse_args(argv)
    commands = build_commands(args)
    for index, command in enumerate(commands, 1):
        display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        print(f"[{index}/{len(commands)}] {display}", flush=True)
    if args.dry_run:
        return 0
    preflight(args)
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Training not started: {error}", file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as error:
        print(f"A codec run failed (exit {error.returncode}); remaining parts were not started.", file=sys.stderr)
        raise SystemExit(error.returncode)
