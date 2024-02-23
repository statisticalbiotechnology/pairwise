from copy import deepcopy
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
import numpy as np
import random
from lance_data_module import LanceDataModule
from parse_args import parse_args_and_config, create_output_dirs
import time

from wrappers.downstream_wrappers import DeNovoRandom, DeNovoTeacherForcing
from wrappers.pretrain_wrappers import (
    MaskedAutoencoderWrapper,
    MaskedTrainingPLWrapper,
    TrinaryMZPLWrapper,
)


import utils

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
    "masked_ae": MaskedAutoencoderWrapper,
    "trinary_mz": TrinaryMZPLWrapper,
}

DOWNSTREAM_TASK_DICT = {
    "denovo_tf": DeNovoTeacherForcing,
    "denovo_random": DeNovoRandom,
}


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)


def main(global_args, pretrain_config=None, ds_config=None):
    print(f"Saving checkpoints in {global_args.output_dir}")
    print(f"Saving logs in {global_args.log_dir}")

    config = {
        **vars(global_args),
        "downstream_config": ds_config,
        "pretrain_config": pretrain_config,
    }

    if global_args.subset:
        if global_args.downstream_task != "none":
            config["downstream_config"][global_args.downstream_task]["subset"] = global_args.subset
        config["pretrain_config"][global_args.pretraining_task]["subset"] = global_args.subset

    # Wandb stuff
    run = None
    logger = None
    if global_args.log_wandb and utils.get_rank() == 0:
        run = wandb.init(
            project=global_args.wandb_project,
            entity=global_args.wandb_entity,
            config=config,
            dir=global_args.log_dir,
            anonymous="allow",
        )
        # this step is for automated hparam sweeping
        update_args(global_args, dict(run.config))
        config = dict(run.config)
        logger = WandbLogger()


    # Define encoder model
    encoder = ENCODER_DICT[global_args.encoder_model](
        use_charge=global_args.use_charge,
        use_mass=global_args.use_mass,
        use_energy=global_args.use_energy,
    )

    if global_args.pretraining_task not in PRETRAIN_TASK_DICT:
        raise NotImplementedError(
            f"{global_args.pretraining_task} pretraining task not implemented"
        )

    distributed = global_args.num_devices > 1 or global_args.num_nodes > 1
    if global_args.pretrain:
        pretrain_data_module = utils.get_lance_data_module(
            global_args.data_root_dir, 
            config["pretrain_config"][global_args.pretraining_task]["batch_size"], 
            global_args.max_peaks,
        )
        pretrain_callbacks = utils.configure_callbacks(
            global_args, 
            config['pretrain_config'][global_args.pretraining_task],
            global_args.pretraining_task + "_val_loss_epoch"
        )

        # Instantiate PL wrapper based on the pretraining task
        pl_encoder = PRETRAIN_TASK_DICT[global_args.pretraining_task](
            encoder,
            global_args=global_args,
            task_dict=config["pretrain_config"][global_args.pretraining_task],
        )

        if run is not None and utils.get_rank() == 0:
            if global_args.watch_model:
                run.watch(pl_encoder, log="all")
            run.log(
                {
                    "num_parameters_encoder": utils.get_num_parameters(encoder),
                }
            )

        (
            print(
                f"Starting distributed pretraining using {global_args.num_devices} devices on {global_args.num_nodes} node(s)"
            )
            if distributed
            else print("Starting single-device training")
        )

        # Define trainer
        pretrainer = pl.Trainer(
            # Distributed kwargs
            accelerator=global_args.accelerator,
            devices=(
                [i for i in range(global_args.num_devices)]
                if global_args.accelerator == "gpu"
                else global_args.num_devices
            ),
            num_nodes=global_args.num_nodes,
            strategy=global_args.strategy if distributed else "auto",
            precision=global_args.precision,
            # Training args
            max_epochs=config["pretrain_config"][global_args.pretraining_task]["epochs"],
            gradient_clip_val=global_args.clip_grad,
            logger=logger,
            callbacks=pretrain_callbacks,
            benchmark=True,
            default_root_dir=global_args.log_dir,
            # profiler="simple",
            barebones=global_args.barebones,
            num_sanity_val_steps=0,
            # detect_anomaly=True,
        )

        if global_args.resume:
            print(f"Resuming training from trainer state: {global_args.encoder_weights}")

        start_time = time.time()
        # This is the call to start training the model
        pretrainer.fit(
            pl_encoder,
            datamodule=pretrain_data_module,
            ckpt_path=global_args.encoder_weights if global_args.resume else None,
        )
        end_time = time.time()  # End time measurement
        print(f"Pretraining finished in {end_time - start_time} seconds")

        # If we keep track of the best model wrt. val loss, select that model and evaluate it on the test set
        if global_args.save_top_k > 0 and global_args.pretrain and not global_args.barebones:
            pretrainer.test(datamodule=pretrain_data_module, ckpt_path="best")

    elif global_args.encoder_weights:
        pl_encoder = PRETRAIN_TASK_DICT[global_args.pretraining_task].load_from_checkpoint(
            global_args.encoder_weights,
            global_args=global_args,
            encoder=encoder,
            task_dict=config["pretrain_config"][global_args.pretraining_task],
        )
        print(f"Loading encoder checkpoint: {global_args.encoder_weights}")
    else:
        print("Warning: proceeding with untrained encoder")

    # ----------- Downstream Finetuning -----------
    # ---------------------------------------------

    # Load best encoder if pretraining has been done
    if global_args.pretrain:
        ckpt_str = (
            "best_model_path"
            if global_args.downstream_encoder == "best"
            else "last_model_path"
        )
        encoder_path = pretrainer.checkpoint_callback.state_dict()[ckpt_str]
        encoder_ckpt = torch.load(encoder_path)
        pl_encoder.load_state_dict(encoder_ckpt["state_dict"])

    if global_args.downstream_task != "none":
        # Extract pretrained encoder nn.Module
        if global_args.pretrain or global_args.encoder_weights:
            encoder = pl_encoder.encoder

        ds_callbacks = utils.configure_callbacks(
            global_args,
            config['downstream_config'][global_args.downstream_task],
            global_args.downstream_task + "_val_aa_prec_epoch", 
            metric_mode="max"
        )

        # Load downstream dataset
        datasets_ds, collate_fn_ds, token_dicts = utils.get_ninespecies_dataset_splits(
            global_args.downstream_root_dir,
            config["downstream_config"],
            max_peaks=global_args.max_peaks,
            subset=config["downstream_config"][global_args.downstream_task]["subset"],
            include_hidden=global_args.downstream_task == "denovo_random",
        )
        # Define decoder model
        assert (
            global_args.decoder_model
        ), f"argument decoder_model must be provided when downstream finetuning"
        decoder = DECODER_DICT[global_args.decoder_model](
            token_dicts, d_model=encoder.running_units
        )

        pl_downstream = DOWNSTREAM_TASK_DICT[global_args.downstream_task](
            encoder,
            decoder,
            global_args=global_args,
            datasets=datasets_ds,
            collate_fn=collate_fn_ds,
            token_dicts=token_dicts,
            task_dict=config["downstream_config"][global_args.downstream_task],
        )

        if run is not None and utils.get_rank() == 0:
            if global_args.watch_model:
                run.watch(pl_downstream, log="all")
            run.log({"num_parameters_decoder": utils.get_num_parameters(decoder)})

        if global_args.downstream_weights:
            print(
                f"Resuming downstream from previous checkpoint: {global_args.downstream_weights}"
            )
            downstream_ckpt = torch.load(
                global_args.downstream_weights, map_location=pl_downstream.device
            )
            pl_downstream.load_state_dict(downstream_ckpt["state_dict"])

        else:
            print(f"Downstream training from scratch")

        (
            print(
                f"Starting distributed downstream finetuning using {global_args.num_devices} devices on {global_args.num_nodes} node(s)"
            )
            if distributed
            else print("Starting single-device training")
        )
        ds_trainer = pl.Trainer(
            # Distributed kwargs
            accelerator=global_args.accelerator,
            devices=(
                [i for i in range(global_args.num_devices)]
                if global_args.accelerator == "gpu"
                else global_args.num_devices
            ),
            num_nodes=global_args.num_nodes,
            strategy=global_args.strategy if distributed else "auto",
            precision=global_args.precision,
            # Training args
            max_epochs=config["downstream_config"][global_args.downstream_task]["epochs"],
            gradient_clip_val=global_args.clip_grad,
            logger=logger,
            callbacks=ds_callbacks,
            benchmark=True,
            default_root_dir=global_args.log_dir,
            # profiler="simple",
            # profiler="advanced",
            barebones=global_args.barebones,
            # num_sanity_val_steps=0,
        )

        start_time = time.time()
        # This is the call to start training the model
        ds_trainer.fit(pl_downstream)
        end_time = time.time()  # End time measurement
        print(f"Downstream finetuning finished in {end_time - start_time} seconds")

        # If we keep track of the best model wrt. val loss, select that model and evaluate it on the test set
        if global_args.save_top_k > 0 and not global_args.barebones:
            ds_trainer.test(ckpt_path="best")

    # Flag the run as finished to the wandb server
    if run is not None and utils.get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    # parse args
    global_args, pretrain_config, ds_config = parse_args_and_config()
    # create output dirs on main process
    create_output_dirs(global_args, is_main_process=utils.get_rank() == 0)
    # A100 specific setting
    if global_args.matmul_precision:
        torch.set_float32_matmul_precision(global_args.matmul_precision)
    # set seed
    torch.manual_seed(global_args.seed)
    np.random.seed(global_args.seed)
    random.seed(global_args.seed)
    pl.seed_everything(global_args.seed)
    # run
    main(global_args, pretrain_config, ds_config)
