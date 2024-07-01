# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Union

import torch
from torch.utils.tensorboard import SummaryWriter
from util import lr_sched, misc


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args, args.lr)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# Copyright contributors to the neural-embedding-compression project
def train_one_epoch_compress(model: Union[torch.nn.parallel.DistributedDataParallel, torch.nn.Module], model_without_ddp: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, aux_optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler: misc.NativeScalerWithGradNormCount,
                    log_writer: SummaryWriter=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    accum_iter = args.accum_iter
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args, args.lr)
            lr_sched.adjust_learning_rate(aux_optimizer, data_iter_step / len(data_loader) + epoch, args, args.prob_lr)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            forward_output = model(samples, mask_ratio=args.mask_ratio)
            loss = forward_output['loss']["loss"]
            recon_loss = forward_output['loss']["recon_loss"]
            bits_loss = forward_output['loss']["bits_loss"]
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(f"Loss: {loss}")
            print(f"Recon Loss: {recon_loss}")
            print(f"Bits Loss: {bits_loss}")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=optimizer.param_groups[0]['params'],
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        with torch.cuda.amp.autocast():
            aux_loss = model_without_ddp.aux_loss()
        
        aux_loss /= accum_iter
        loss_scaler(aux_loss, aux_optimizer, parameters=aux_optimizer.param_groups[0]['params'],
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

        torch.cuda.synchronize()

        recon_loss_value = recon_loss.item()
        bits_loss_value = bits_loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(recon_loss=recon_loss_value)
        metric_logger.update(bits_loss=bits_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value, device=device)
        recon_loss_value_reduce = misc.all_reduce_mean(recon_loss_value, device=device)
        bits_loss_value_reduce = misc.all_reduce_mean(bits_loss_value, device=device)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('recon_loss', recon_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('bits_loss', bits_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            #log images
            if epoch % 2 == 0 and data_iter_step == 0:
                # unnormalize
                log_writer.add_images('original_images', unnormalize(samples[:16, ...]), epoch_1000x)
                ## the model does patching wrong! but we have to keep it wrong to make it happy...
                ## I dont think it has a negative effect on performance though
                preds = model_without_ddp.unpatchify(forward_output['pred'][:16, ...]) 
                log_writer.add_images('reconstructed_images', unnormalize(preds), epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes(device=device)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def unnormalize(x):
    return x * x.new([0.229, 0.224, 0.225]).view(3, 1, 1) + x.new([0.485, 0.456, 0.406]).view(3, 1, 1)