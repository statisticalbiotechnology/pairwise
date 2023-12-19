import argparse
import os
import datetime
import yaml


def get_args_parser(conf_parser):
    parser = argparse.ArgumentParser(
        "Train unsupervised transformers", parents=[conf_parser]
    )
    # Model parameters
    parser.add_argument(
        "--encoder_model",
        default="abc",
        type=str,
        help="Name of the encoder model to train",
    )
    parser.add_argument(
        "--decoder_model",
        default="",
        type=str,
        help="Name of the encoder model to train",
    )
    parser.add_argument(
        "--pretraining_task",
        default="masked",
        choices=["masked", "trinary_mz", "dummy", "denovo"],
        type=str,
        help="Which pretraining strategy to use",
    )
    parser.add_argument(
        "--input_mass",
        default=0,
        type=bool,
        help="1: input precursor mass",
    )
    parser.add_argument(
        "--input_charge",
        default=0,
        type=bool,
        help="1: input precursor charge",
    )
    parser.add_argument(
        "--mask_zero_tokens",
        default=0,
        type=bool,
        help="1: mask the attention for 'null' tokens",
    )
    parser.add_argument(
        "--trinary_freq",
        default=0.15,
        type=float,
        help="Frequency of corrupted tokens in the Trinary MZ objective",
    )
    parser.add_argument(
        "--trinary_std",
        default=5,
        type=float,
        help="Stdev of Gaussian noise in the corrupted tokens in the Trinary MZ objective",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--max_peaks",
        default=300,
        type=int,
        help="The maximally allowed number of peaks. Spectra that have more peaks are subsampled by the maximum intensities.",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed tokens).",
    )
    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-4,
        help="base learning rate (max lr)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-07,
        help="lower lr bound for cyclic schedulers",
    )
    parser.add_argument(
        "--anneal_lr",
        type=bool,
        default=True,
        help="Turn on cosine annealing lr",
    )
    parser.add_argument(
        "--scale_lr_by_batchsize",
        type=bool,
        default=False,
        help="Turns on MAE-style lr scaling which multiplies lr by (eff_batch_size / 256)",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_root_dir",
        default="../../datasets/instanovo_data_subset",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save",
    )
    parser.add_argument("--log_dir", default="./log_dir", help="path where to log")
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=1,
        help="saves top-K checkpoints based on 'val_loss' metric",
    )
    parser.add_argument(
        "--save_last",
        type=bool,
        default=True,
        help="saves checkpoint from last epoch",
    )
    parser.add_argument(
        "--every_n_epochs",
        type=int,
        default=None,
        help="Number of epochs between checkpoints",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=0,
        help="(int) If >0, early stop with patience = this int",
    )
    parser.add_argument("--seed", default=0, type=int)
    # parser.add_argument("--resume", default="", help="resume from checkpoint")
    # parser.add_argument(
    #     "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    # )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        type=bool,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    # distributed training parameters
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["cpu", "gpu", "mps"],
        default="gpu",
        help="Specify the accelerator type",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["16-mixed", "bf16-mixed", "32-true", "64-true"],
        default="32-true",
        help=" Double precision, full precision (32), 16bit mixed precision or bfloat16 mixed precision",
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        choices=["medium", "high", None],
        default=None,
        help="To fully exploit NVIDIA A100 GPUs, set torch.set_float32_matmul_precision('medium' | 'high') which trades off precision for performance",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["ddp", "deepspeed"],
        default="ddp",
        help="Specify the distributed strategy",
    )
    parser.add_argument(
        "--num_devices", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes")

    parser.add_argument("--clip_grad", type=float, default=None, help="")
    parser.add_argument(
        "--log_wandb", default=1, type=int, help="Disable WandB logging by setting to 0"
    )
    parser.add_argument(
        "--wandb_project", default=None, help="Specify project name to log using WandB"
    )
    parser.add_argument(
        "--wandb_entity", default="kall", help="Entity to log as on WandB"
    )
    parser.add_argument(
        "--loss_type",
        choices=["mse", "ce", "bce"],
        default="mse",
        help="MSE, CE or Binary CE (latter only makes sense for binary targets)",
    )
    parser.add_argument(
        "--profile_flops",
        type=bool,
        default=False,
        help="Measure forward pass FLOPs on the first train batch",
    )
    return parser


def sanity_checks(args):
    # add sanity checks, i.e. for args that should be mutually exclusive here
    args.log_wandb = bool(args.log_wandb)
    pass


def uniquify_path(path):
    # append a number
    counter = 1
    try_path = path
    if not os.path.exists(try_path):
        return path
    else:
        while os.path.exists(try_path):
            try_path = path + "__" + str(counter)
            counter += 1

        return try_path


def parse_args_and_config():
    ### Priority: provided command line args > config values > argparse defaults

    # parse the config arg only
    conf_parser = argparse.ArgumentParser("Config parser", add_help=False)
    conf_parser.add_argument(
        "--config",
        type=str,
        help="config path",
    )
    conf_args, remaining_args = conf_parser.parse_known_args()

    # open config file and set default args to those included in the config
    try:
        with open(conf_args.config, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print("Error occurred while loading the configuration file:")
        print(e)

    parser = get_args_parser(conf_parser)
    parser.set_defaults(**config)

    # parse the rest of the args and override defaults/config
    args = parser.parse_args(remaining_args)
    sanity_checks(args)
    return args


def create_output_dirs(args, is_main_process=True):
    if is_main_process:
        # append datetime str to output dirs
        now = datetime.datetime.now()
        cur_time = now.strftime("_%H_%M_%S_%f__%d_%m_%y")
        args.output_dir = args.output_dir + cur_time
        args.output_dir = uniquify_path(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=False)
        args.log_dir = args.log_dir + cur_time
        args.log_dir = uniquify_path(args.log_dir)
        os.makedirs(args.log_dir, exist_ok=False)


if __name__ == "__main__":
    # test code
    args = parse_args_and_config()
    # create_output_dirs(args)
