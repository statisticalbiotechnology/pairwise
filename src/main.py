import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import numpy as np
import random
from parse_args import parse_args_and_config, create_output_dirs


from pl_callbacks import FLOPProfilerCallback, CosineAnnealLRCallback
from pl_wrappers import ReconstructionPLWrapper

import models.encoders.dummy_encoders as encoders
import models.heads.dummy_heads as heads

ENCODER_DICT = {
    **encoders.__dict__, #TODO: replace with actual models
}
HEAD_DICT = {
    **heads.__dict__, #TODO: replace with actual models
}

def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)


def main(args):
    print(f"Saving checkpoints in {args.output_dir}")
    print(f"Saving logs in {args.log_dir}")

    # Wandb stuff
    run = None
    logger = None
    if args.log_wandb and utils.get_rank() == 0:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            dir=args.output_dir,
            anonymous="allow"
        )
        # this step is for automated hparam sweeping
        update_args(
            args, dict(run.config)
        )
        logger = WandbLogger()

    # lr scaling by batch size trick
    eff_batch_size = (
        args.batch_size * args.accum_iter * args.num_devices * args.num_nodes
    )
    if args.scale_lr_by_batchsize:
        args.lr = args.blr * eff_batch_size / 256 if args.lr is None else args.lr
    else:
        args.lr = args.blr

    callbacks = []
    # Checkpoint callback
    callbacks += [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",  # requires that we log something called "val_loss"
            mode="min",
            save_top_k=args.save_top_k,
            save_last=args.save_last,
            every_n_epochs=args.every_n_epochs,
        )
    ]

    if args.anneal_lr:
        # Cosine annealing LR with warmup callback
        callbacks += [
            CosineAnnealLRCallback(
                lr=args.lr, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs
            )
        ]

    # measure FLOPs on the first train batch
    if args.profile_flops:
        callbacks += [FLOPProfilerCallback()]

    datasets = ...  # utils.get_dataset(args) TODO: implement Datasets

    # Define encoder model
    encoder = ENCODER_DICT[args.encoder_model](
        args.example_arg1,
        args.example_arg2,
        args.example_arg3,
        args.example_arg4,
    )
    # Define head model
    head = HEAD_DICT[args.head_model](
        args.example_arg5,
        args.example_arg6,
        args.example_arg7,
        args.example_arg8,
    )

    if args.loss_type == "mse":
        loss_fn = torch.nn.MSELoss(reduction="mean")
    elif args.loss_type == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    elif args.loss_type == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    else:
        raise NotImplementedError(f"Invalid loss func {args.loss_type} specified")

    pl_model = ReconstructionPLWrapper(encoder, head, loss_fn=loss_fn, args=args, datasets=datasets)

    if args.early_stop > 0:
        callbacks += [EarlyStopping("val_loss", patience=args.early_stop)]

    if run is not None and utils.get_rank() == 0:  # TODO: implement get_rank
        run.log({"num_parameters": utils.get_num_parameters(model)}) # TODO: implement get_num_parameters

    # Define trainer
    distributed = args.num_devices > 1 or args.num_nodes > 1
    print(
        f"Starting distributed training using {args.num_devices} devices on {args.num_nodes} node(s)"
    ) if distributed else print("Starting single-device training")

    trainer = pl.Trainer(
        # Distributed kwargs
        accelerator=args.accelerator,
        devices=[i for i in range(args.num_devices)]
        if args.accelerator == "gpu"
        else args.num_devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy if distributed else "auto",
        precision=args.precision,
        # Training args
        max_epochs=args.epochs,
        gradient_clip_val=args.clip_grad,
        logger=logger,
        callbacks=callbacks,
        benchmark=True,
        # profiler="simple"
    )

    # This is the call to start training the model
    trainer.fit(pl_model)

    # If we keep track of the best model wrt. val loss, select that model and evaluate it on the test set
    if args.save_top_k > 0:
        trainer.test(ckpt_path="best")

    # Flag the run as finished to the wandb server
    if run is not None and utils.get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    # parse args
    args = parse_args_and_config()
    # create output dirs on main process
    create_output_dirs(args, is_main_process=utils.get_rank() == 0)
    # A100 specific setting
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # run
    main(args)
