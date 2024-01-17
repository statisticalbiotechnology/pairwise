import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import numpy as np
import random
from parse_args import parse_args_and_config, create_output_dirs
import time

from pl_callbacks import FLOPProfilerCallback, CosineAnnealLRCallback
from pl_wrappers import (
    DeNovoPLWrapper,
    DummyPLWrapper,
    MaskedTrainingPLWrapper,
    TrinaryMZPLWrapper,
)
import utils
from loader_parquet import PeptideParser

import models.encoder as encoders
import models.dc_encoder as dc_encoders
import models.decoder as decoders
import models.dc_decoder as dc_decoders

ENCODER_DICT = {
    **encoders.__dict__,
    **dc_encoders.__dict__,
}
DECODER_DICT = {
    **decoders.__dict__,
    **dc_decoders.__dict__,
}

PRETRAIN_TASK_DICT = {
    "masked": MaskedTrainingPLWrapper,
    "dummy": DummyPLWrapper,
    "trinary_mz": TrinaryMZPLWrapper,
}

DOWNSTREAM_TASK_DICT = {
    "denovo": DeNovoPLWrapper,
}


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)


def main(args, ds_config=None):
    print(f"Saving checkpoints in {args.output_dir}")
    print(f"Saving logs in {args.log_dir}")

    config = {**vars(args), "downstream": ds_config}
    # Wandb stuff
    run = None
    logger = None
    if args.log_wandb and utils.get_rank() == 0:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
            dir=args.log_dir,
            anonymous="allow",
        )
        # this step is for automated hparam sweeping
        update_args(args, dict(run.config))
        config = dict(run.config)
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
    if not args.barebones:
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

    datasets, collate_fn = utils.get_spectrum_dataset_splits(
        args.data_root_dir,
        splits=[0.7, 0.2, 0.1],
        max_peaks=args.max_peaks,
        subset=args.subset,
    )

    # Define encoder model
    encoder = ENCODER_DICT[args.encoder_model](
        use_charge=args.use_charge,
        use_mass=args.use_mass,
        use_energy=args.use_energy,
    )

    if args.pretraining_task not in PRETRAIN_TASK_DICT:
        raise NotImplementedError(
            f"{args.pretraining_task} pretraining task not implemented"
        )

    distributed = args.num_devices > 1 or args.num_nodes > 1
    if args.pretrain:
        # Instantiate PL wrapper based on the pretraining task
        pl_encoder = PRETRAIN_TASK_DICT[args.pretraining_task](
            encoder, args=args, datasets=datasets, collate_fn=collate_fn
        )

        if args.early_stop > 0:
            callbacks += [EarlyStopping("val_loss", patience=args.early_stop)]

        if run is not None and utils.get_rank() == 0:
            run.watch(pl_encoder, "all")
            run.log(
                {
                    "num_parameters_encoder": utils.get_num_parameters(encoder),
                }
            )

        print(
            f"Starting distributed pretraining using {args.num_devices} devices on {args.num_nodes} node(s)"
        ) if distributed else print("Starting single-device training")

        # Define trainer
        pretrainer = pl.Trainer(
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
            default_root_dir=args.log_dir,
            # profiler="simple",
            barebones=args.barebones,
        )

        if args.resume:
            print(f"Resuming training from trainer state: {args.encoder_weights}")

        start_time = time.time()
        # This is the call to start training the model
        pretrainer.fit(
            pl_encoder, ckpt_path=args.encoder_weights if args.resume else None
        )
        end_time = time.time()  # End time measurement
        print(f"Pretraining finished in {end_time - start_time} seconds")

        # If we keep track of the best model wrt. val loss, select that model and evaluate it on the test set
        if args.save_top_k > 0 and args.pretrain and not args.barebones:
            pretrainer.test(ckpt_path="best")

    elif args.encoder_weights:
        pl_encoder = PRETRAIN_TASK_DICT[args.pretraining_task].load_from_checkpoint(
            args.encoder_weights, args=args, encoder=encoder, datasets=datasets
        )
        print(f"Loading encoder checkpoint: {args.encoder_weights}")
    else:
        print("Warning: proceeding with untrained encoder")

    # ----------- Downstream Finetuning -----------
    # ---------------------------------------------

    # Load best encoder if pretraining has been done
    if args.pretrain:
        ckpt_str = (
            "best_model_path"
            if args.downstream_encoder == "best"
            else "last_model_path"
        )
        encoder_path = pretrainer.checkpoint_callback.state_dict()[ckpt_str]
        encoder_ckpt = torch.load(encoder_path)
        pl_encoder.load_state_dict(encoder_ckpt["state_dict"])

    datasets_ds, collate_fn_ds, token_dicts = utils.get_ninespecies_dataset_splits(
        args.downstream_root_dir,
        ds_config,
        max_peaks=args.max_peaks,
        subset=args.subset,
    )

    if args.downstream_task != "none":
        # Extract pretrained encoder nn.Module
        if args.pretrain or args.encoder_weights:
            encoder = pl_encoder.encoder

        # Define decoder model
        assert (
            args.decoder_model
        ), f"argument decoder_model must be provided when downstream finetuning"
        decoder = DECODER_DICT[args.decoder_model](
            token_dicts, d_model=encoder.running_units
        )

        if run is not None and utils.get_rank() == 0:
            run.log({"num_parameters_decoder": utils.get_num_parameters(decoder)})

        pl_downstream = DOWNSTREAM_TASK_DICT[args.downstream_task](
            encoder,
            decoder,
            args=args,
            datasets=datasets_ds,
            collate_fn=collate_fn_ds,
            token_dicts=token_dicts,
            conf_threshold=config["downstream"]["conf_threshold"],
        )

        print(
            f"Starting distributed downstream finetuning using {args.num_devices} devices on {args.num_nodes} node(s)"
        ) if distributed else print("Starting single-device training")
        ds_trainer = pl.Trainer(
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
            default_root_dir=args.log_dir,
            # profiler="simple",
            # profiler="advanced",
            barebones=args.barebones,
            # num_sanity_val_steps=0,
        )

        start_time = time.time()
        # This is the call to start training the model
        ds_trainer.fit(pl_downstream)
        end_time = time.time()  # End time measurement
        print(f"Downstream finetuning finished in {end_time - start_time} seconds")

        # If we keep track of the best model wrt. val loss, select that model and evaluate it on the test set
        if args.save_top_k > 0 and not args.barebones:
            ds_trainer.test(ckpt_path="best")

    # Flag the run as finished to the wandb server
    if run is not None and utils.get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    # parse args
    args, ds_config = parse_args_and_config()
    # create output dirs on main process
    create_output_dirs(args, is_main_process=utils.get_rank() == 0)
    # A100 specific setting
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    pl.seed_everything(args.seed)
    # run
    main(args, ds_config)
