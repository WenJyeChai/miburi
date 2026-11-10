import math

import torch


def create_scheduler(args, optimizer, **kwargs):
    num_epochs = args.epochs

    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if args.lr_policy == "onecyclelr":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr_base,
            total_steps=kwargs["total_steps"],
            pct_start=0.05,
            # div_factor=args.DIV_FACTOR_ONECOS,
            # final_div_factor=args.FIN_DACTOR_ONCCOS,
        )
    elif args.lr_policy == "cycliclr":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr_base / 10,
            max_lr=args.lr_base,
            step_size_up=args.lr_cyclestepsizeup,
            mode="triangular2",
            cycle_momentum=False,
        )
    elif args.lr_policy == "cosinerestart":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = 1,
            T_mult=2,
            eta_min = 1e-8,
            last_epoch=-1,
        )
    elif args.lr_policy == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.decay_epochs,
            gamma=args.decay_rate,
        )
    elif args.lr_policy == "cosine_delay":
        # Linear warmup 0 -> lr_base over [0, warmup_epochs) if warmup_epochs
        # > 0 (reuses the previously-unused --warmup_epochs; --warmup_lr is
        # NOT used here -- the warmup peak is always lr_base, matching "warm
        # up to lr_base, then decay to lr_min" rather than overshooting past
        # lr_base). Decay then always starts right where warmup ends
        # (lr_cosine_start_epoch is only meaningful when warmup_epochs == 0,
        # for a flat-then-decay shape with no ramp-up).
        warmup_epochs = int(getattr(args, "warmup_epochs", 0) or 0)
        start_epoch = max(warmup_epochs, int(args.lr_cosine_start_epoch))
        end_epoch = int(args.lr_cosine_end_epoch)
        if end_epoch < 0:
            end_epoch = num_epochs
        if end_epoch <= start_epoch:
            raise ValueError(
                "lr_cosine_end_epoch must be greater than "
                "max(warmup_epochs, lr_cosine_start_epoch) (got "
                f"start={start_epoch}, end={end_epoch})."
            )
        eta_min_ratio = args.lr_min / args.lr_base

        def _cosine_delay_factor(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max(0.0, epoch / warmup_epochs)
            if epoch <= start_epoch:
                return 1.0
            progress = min(1.0, (epoch - start_epoch) / (end_epoch - start_epoch))
            return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_cosine_delay_factor,
        )
    else:
        raise ValueError(f"Unknown LR policy: {args.lr}")
    
    return lr_scheduler