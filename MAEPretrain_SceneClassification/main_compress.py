# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np

# assert timm.__version__ == "0.3.2"  # version check
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import util.misc as misc
from engine_pretrain import train_one_epoch_compress
from models_mae import mae_vit_compress_adapter, mae_vit_compress_hyperprior_adapter
from PIL import Image
from timm.data import create_dataset
from torch.distributed import destroy_process_group
from torch.utils.tensorboard import SummaryWriter
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from MAEPretrain_SceneClassification.util.compression import (
    findEmbeddingSize,
    net_aux_optimizer,
)

Image.MAX_IMAGE_PIXELS = None

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training")
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
        help="Batch size per GPU for eval.(effective batch size is batch_size * accum_iter * # gpus). For the final eval, only 1 gpu is used, so the effective batch size is the same as this)",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--save_every_n_epochs", default=100, type=int)
    parser.add_argument("--trainable_blocks", default=1, type=int, help="number of blocks in encoder and decoder which are trained")

    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_compress_hyperprior_adapter",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument("--ld", required=True, type=float, help="lambda for R-D loss")

    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )

    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument("--finetune", required=True, help="finetune from checkpoint")

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--prob_lr", type=float, default=None, metavar="LR", help="lr for prob model"
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument("--data_path", required=True, type=str, help="dataset path")

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--eval_size",
        default=1000,
        type=int,
        help="Number of samples to use to estimate embedding size",
    )

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # dataset
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        choices=["millionAID", "imagenet"],
        help="type of dataset",
    )

    # gpu_num
    parser.add_argument("--gpu_num", default=None, type=int, help="number of gpus")

    parser.add_argument(
        "--tag", default=None, type=int, help="different number of training samples"
    )

    return parser


def main(args):
    args.distributed = "LOCAL_RANK" in os.environ

    if args.distributed:
        dist.init_process_group("nccl")
        args.gpu_num = dist.get_world_size()
        print(
            f"Local Rank: {os.environ['LOCAL_RANK']}, Global Rank: {os.environ['RANK']}"
        )
        rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        torch.cuda.set_device(rank)
        misc.setup_for_distributed(global_rank == 0)
    else:
        args.gpu_num = torch.cuda.device_count()
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    if args.distributed:
        device = torch.device(rank)
    else:
        device = torch.device("cuda")

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.2, 1.0), interpolation=3
            ),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.2, 1.0), interpolation=3
            ),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.dataset == "imagenet":
        dataset_train = datasets.ImageFolder(
            os.path.join(args.data_path, "train"), transform=transform_train
        )

    elif args.dataset == "millionAID":
        # from the download, the test directory is by far the larger one. They may be mislabeled. Use `test` for training
        dataset_train = create_dataset(
            "MillionAid",
            os.path.join(args.data_path, "test"),
            transform=transform_train,
            is_training=True,
        )
        dataset_eval = create_dataset(
            "MillionAid",
            os.path.join(args.data_path, "train"),
            transform=transform_test,
        )
        if len(dataset_eval) > args.eval_size:
            dataset_eval = torch.utils.data.Subset(
                dataset_eval, torch.arange(args.eval_size)
            )
        else:
            args.eval_size = len(dataset_eval)

    else:
        raise NotImplementedError

    # output folder
    args.output_dir = os.path.join(
        args.output_dir,
        args.dataset + "_" + str(args.input_size),
        str(args.epochs)
        + "_"
        + str(args.mask_ratio)
        + "_"
        + str(args.blr)
        + "_"
        + str(args.weight_decay)
        + "_"
        + str(args.batch_size * args.gpu_num),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Length of train dataset: {len(dataset_train)}")
    print(f"Length of eval dataset: {len(dataset_eval)}")

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
        sampler_eval = torch.utils.data.DistributedSampler(dataset_eval, shuffle=False)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_eval = torch.utils.data.RandomSampler(dataset_eval)

    if (not args.distributed or global_rank == 0) and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=False,
    )
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        sampler=sampler_eval,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    # define the model

    if args.model == "mae_vit_compress_hyperprior_adapter":
        model = mae_vit_compress_hyperprior_adapter(args.ld, norm_pix_loss=args.norm_pix_loss, trainable_encoder_decoder_blocks=args.trainable_blocks)
    elif args.model == "mae_vit_compress_adapter":
        model = mae_vit_compress_adapter(args.ld, norm_pix_loss=args.norm_pix_loss, trainable_encoder_decoder_blocks=args.trainable_blocks)
    else:
        raise Exception("Unknown model")

    checkpoint = torch.load(args.finetune, map_location="cpu")

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint["model"]
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    if args.prob_lr is None:
        args.prob_lr = args.lr

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False
        )
        model_without_ddp = model.module

    def configure_optimizers(net, lr, prob_lr):
        """Separate parameters for the main optimizer and the auxiliary optimizer.
        Return two optimizers"""
        conf = {
            "net": {"type": "AdamW", "lr": lr, "betas": (0.9, 0.95)},
            "aux": {"type": "Adam", "lr": prob_lr},
        }
        optimizer = net_aux_optimizer(net, conf)
        return optimizer["net"], optimizer["aux"]

    optimizer, aux_optimizer = configure_optimizers(model, args.lr, args.prob_lr)
    print(optimizer, aux_optimizer)
    loss_scaler = NativeScaler(enabled=True)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch_compress(
            model,
            model_without_ddp,
            data_loader_train,
            optimizer,
            aux_optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir:
            if epoch % args.save_every_n_epochs == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                latest=True,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    print("Estimating average embedding size...")
    model_without_ddp.update()
    # do this without ddp

    dataloader = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    if not args.distributed or global_rank == 0:
        embedding_sizes = findEmbeddingSize(model_without_ddp, dataloader)
        av_size = (embedding_sizes.sum() / args.eval_size) * 8
        print(f"Average bits per image: {av_size}")
        log_writer.add_scalar("Average bits", av_size)

    if args.distributed:
        destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
