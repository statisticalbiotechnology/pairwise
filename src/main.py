from copy import deepcopy
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
import numpy as np
import random
from data.lance_data_module import LanceDataModule
from parse_args import parse_args_and_config, create_output_dirs
import time
import shutil

from wrappers.downstream_wrappers import DeNovoTeacherForcing
from wrappers.casanovo_trainer_wrapper import DeNovoSpec2Pep

import utils

import models.encoder as encoders
import models.dc_encoder as dc_encoders
import models.decoder as decoders
import models.dc_decoder as dc_decoders
import models.casanovo.encoder_interface as casanovo_encoders
import models.casanovo.decoder_interface as casanovo_decoders

ENCODER_DICT = {
    **encoders.__dict__,
    **dc_encoders.__dict__,
    **casanovo_encoders.__dict__,
}
DECODER_DICT = {
    **decoders.__dict__,
    **dc_decoders.__dict__,
    **casanovo_decoders.__dict__,
}

TRAINING_WRAPPERS = {
    "denovo_tf": DeNovoTeacherForcing,
    "casanovo_tf": DeNovoSpec2Pep,
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
        dropout=config["pretrain_config"][global_args.pretraining_task].get(
            "dropout", 0
        ),
    )

    distributed = global_args.num_devices > 1 or global_args.num_nodes > 1

    if run is not None and utils.get_rank() == 0:
        run.log(
            {
                "num_parameters_encoder": utils.get_num_parameters(encoder),
            }
        )

    # ----------- Supervised Training -----------
    # ---------------------------------------------

    if global_args.downstream_task != "none":

        ds_callbacks = utils.configure_callbacks(
            global_args,
            config["downstream_config"][global_args.downstream_task],
            global_args.downstream_task + "_val_loss_epoch",
            metric_mode="min",
        )

        # Load downstream dataset
        if config["downstream_config"]["dataset_name"] == "ninespecies":
            ds_data_module, token_dicts = utils.get_ninespecies_data_module(
                config["downstream_config"],
                global_args,
                # seed=global_args.seed, #TODO: This should accept a seed
            )
        elif config["downstream_config"]["dataset_name"] == "ninespecies_hf":
            ds_data_module, token_dicts = utils.get_ninespecies_HF_data_module(
                config["downstream_config"],
                global_args,
                # seed=global_args.seed, #TODO: This should accept a seed
            )
        elif config["downstream_config"]["dataset_name"] == "massivekb":
            ds_data_module, token_dicts = utils.get_mskb_data_module(
                config["downstream_config"], global_args, seed=global_args.seed
            )

        # Define decoder model
        assert (
            global_args.decoder_model
        ), f"argument decoder_model must be provided when downstream finetuning"
        decoder = DECODER_DICT[global_args.decoder_model](
            token_dicts,
            d_model=encoder.running_units,
            dropout=config["downstream_config"][global_args.downstream_task][
                "decoder_dropout"
            ],
            max_seq_len=global_args.max_length + 1,  # +1 because of added EOS token
        )

        pl_downstream = TRAINING_WRAPPERS[global_args.downstream_task](
            encoder,
            decoder,
            global_args=global_args,
            token_dicts=token_dicts,
            task_dict=config["downstream_config"][global_args.downstream_task],
        )

        if run is not None and utils.get_rank() == 0:
            if global_args.watch_model:
                run.watch(pl_downstream, log="all")
            run.log({"num_parameters_decoder": utils.get_num_parameters(decoder)})

        if global_args.downstream_weights:
            print(
                f"Loading downstream weights from previous checkpoint: {global_args.downstream_weights}"
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
            max_epochs=(
                config["downstream_config"][global_args.downstream_task]["epochs"]
                if global_args.epochs < 1
                else global_args.epochs
            ),
            gradient_clip_val=global_args.clip_grad,
            logger=logger,
            callbacks=ds_callbacks,
            benchmark=True,
            default_root_dir=global_args.log_dir,
            # profiler="simple",
            # profiler="advanced",
            barebones=global_args.barebones,
            # num_sanity_val_steps=0,
            limit_train_batches=config["limit_train_batches"],
            limit_val_batches=config["limit_val_batches"],
            check_val_every_n_epoch=config["validate_every_n_epochs"],
        )

        if global_args.eval_only:
            print("'--eval_only' specified - skipping training")
            # ds_trainer.validate(pl_downstream, datamodule=ds_data_module)
            ds_trainer.test(pl_downstream, datamodule=ds_data_module)
        else:
            if global_args.resume:
                print(
                    f"Resuming training from trainer state: {global_args.downstream_weights}"
                )
            start_time = time.time()
            # This is the call to start training the model
            ds_trainer.fit(
                pl_downstream,
                datamodule=ds_data_module,
                ckpt_path=(
                    global_args.downstream_weights if global_args.resume else None
                ),
            )
            end_time = time.time()  # End time measurement
            print(f"Downstream finetuning finished in {end_time - start_time} seconds")

            # If we keep track of the best model wrt. val loss, select that model and evaluate it on the test set
            if global_args.save_top_k > 0 and not global_args.barebones:
                ds_trainer.test(datamodule=ds_data_module, ckpt_path="best")

    # Flag the run as finished to the wandb server
    if run is not None and utils.get_rank() == 0:
        wandb.finish()

    if global_args.remove_ckpt:
        shutil.rmtree(global_args.output_dir, ignore_errors=True)


if __name__ == "__main__":
    # parse args
    global_args, pretrain_config, ds_config = parse_args_and_config()
    # create output dirs on main process
    create_output_dirs(global_args, is_main_process=utils.get_rank() == 0)
    # A100 specific setting
    if global_args.matmul_precision:
        torch.set_float32_matmul_precision(global_args.matmul_precision)
    # set seed
    if global_args.seed is not None:
        torch.manual_seed(global_args.seed)
        np.random.seed(global_args.seed)
        random.seed(global_args.seed)
        pl.seed_everything(global_args.seed)
    else:
        print("JL - Using a random seed")
    # run
    main(global_args, pretrain_config, ds_config)
