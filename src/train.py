# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/train.py
# @Author: Qinlong Huang
# @Create Date: 2021/03/31 14:34
# @Contact: qinlonghuang@gmail.com
# @Description:

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import losses
import shutil
import math
import utils
import models
from tqdm import tqdm
import time
import os
import sys
from src.data import PriorDataset, collate_tvs
from torch.utils.data import DataLoader

from torch.utils import tensorboard


def train_sleep(generative_prior_model: models.GenerativeModelPriorDistribution,
                generative_condition_model: models.GenerativeModelConditionDistribution,
                inference_model: models.InferenceModel,
                lr,
                num_samples, num_iterations, writer: tensorboard.SummaryWriter):
    optimizer_inference = torch.optim.Adam(inference_model.parameters(), lr=lr)  # sleep阶段只训练识别模型
    device = next(inference_model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device=device)

    utils.logging.info("Pretraining with sleep")
    pbar = tqdm(range(num_iterations), total=num_iterations)
    start_time = time.time()
    sleep_loss = 0.0
    for i in pbar:

        optimizer_inference.zero_grad()

        sleep_loss, sleep_interpolate_time, ce_loss, entropy_loss, non_target_prob = losses.get_sleep_loss(
            generative_prior_model, generative_condition_model,
            inference_model, num_samples)
        forward_time = time.time() - start_time
        sleep_loss.backward()
        optimizer_inference.step()
        backward_time = time.time() - start_time - forward_time

        pbar.set_description("Train sleep efficiency in iteration {}: fw: {:.2f}, bw: {:.2f}, "
                             "sleep loss: {:.6f}, "
                             "GPU memory: {:.2f} MB".format(i, forward_time / (forward_time + backward_time),
                                                            backward_time / (forward_time + backward_time),
                                                            sleep_loss.item(),
                                                            (
                                                                torch.cuda.max_memory_allocated(device=device) / 1e6
                                                                if device.type == "cuda"
                                                                else 0
                                                            )))

        writer.add_scalar('pretrain_sleep_losses/recognition_loss', sleep_loss.item(), global_step=i)
        writer.add_scalar('pretrain_sleep_losses/ce_loss', ce_loss, global_step=i)
        writer.add_scalar('pretrain_sleep_losses/entropy_loss', entropy_loss, global_step=i)
        writer.add_scalar('pretrain_sleep_losses/non_target_prob', non_target_prob, global_step=i)

        start_time = time.time()

    return sleep_loss.item()


def train_ws(
        generative_prior_model: models.GenerativeModelPriorDistribution,
        generative_condition_model: models.GenerativeModelConditionDistribution,
        inference_model: models.InferenceModel,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizer_generative_prior: torch.optim.Optimizer,
        optimizer_generative_condition: torch.optim.Optimizer,
        optimizer_inference: torch.optim.Optimizer,
        writer: tensorboard.SummaryWriter,
        cur_iteration,
        cfg=None,
):
    device = next(inference_model.parameters()).device

    # xs_inference, num_frames_inference = next(iter(train_loader))
    # xs_inference = xs_inference[:cfg.TRAIN.PLOT_SAMPLES]
    # num_frames_inference = num_frames_inference[:cfg.TRAIN.PLOT_SAMPLES]

    eval_condition_loss_meter = utils.AverageMeter()
    eval_reconstruction_loss_meter = utils.AverageMeter()
    eval_diff_loss_meter = utils.AverageMeter()
    eval_prior_loss_meter = utils.AverageMeter()
    eval_recognition_loss_meter = utils.AverageMeter()
    eval_ce_loss_meter = utils.AverageMeter()
    eval_entropy_loss_meter = utils.AverageMeter()

    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device=device)

    print('Starting WS training !')

    for epoch in range(cur_iteration, cfg.TRAIN.MAX_EPOCH):

        inference_model.train()
        generative_condition_model.train()
        generative_prior_model.train()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc='Train epoch {}/{}'.format(epoch, cfg.TRAIN.MAX_EPOCH))
        for i, (xs, num_frames) in pbar:
            start_time = time.time()
            if cfg.TRAIN.DEVICE == 'gpu':
                xs = xs.cuda()  # [batch_size, num_frame, 15]
            prepare_time = time.time() - start_time

            # wake phase
            optimizer_generative_prior.zero_grad()
            optimizer_generative_condition.zero_grad()

            '''
            z_probs: list of ndarray, 每个ndarray代表每一帧在每一个target维度上的概率
            z_preds: list of ndarray, 每个ndarray代表采样后实际z序列
            x_preds: list of ndarray, 每个ndarray代表用z_pred画出来的x_pred
            '''
            condition_loss, reconstruction_loss, diff_loss, prior_loss, wake_interpolate_time, z_probs, z_preds, \
                x_preds = losses.get_wake_loss(generative_prior_model, generative_condition_model, inference_model,
                                               xs, num_frames, num_samples=cfg.TRAIN.NUM_SAMPLES,
                                               plot_samples=cfg.TRAIN.PLOT_SAMPLES)
            prior_loss.backward()

            optimizer_generative_prior.step()
            optimizer_generative_condition.step()

            wake_time = time.time() - start_time - prepare_time

            # train sleep
            optimizer_inference.zero_grad()
            sleep_loss, sleep_interpolate_time, ce_loss, entropy_loss, non_target_prob = losses.get_sleep_loss(
                generative_prior_model, generative_condition_model, inference_model,
                cfg.TRAIN.NUM_SAMPLES * cfg.TRAIN.BATCH_SIZE)
            sleep_loss.backward()
            optimizer_inference.step()
            sleep_time = time.time() - start_time - wake_time - prepare_time

            pbar.set_description("Epoch_iter {}_{}: "
                                 "w_eff: {:.2f}, "
                                 "w_interp_eff: {:.2f}, "
                                 "s_eff: {:.2f}, "
                                 "s_interp_eff: {:.2f}, "
                                 "interp_eff: {:.2f}".format(epoch, i,
                                                             wake_time / (wake_time + sleep_time),
                                                             wake_interpolate_time / (wake_time + sleep_time),
                                                             sleep_time / (wake_time + sleep_time),
                                                             sleep_interpolate_time / (wake_time + sleep_time),
                                                             (wake_interpolate_time + sleep_interpolate_time) /
                                                             (wake_time + sleep_time)
                                                             ))

            # log in tb
            step = epoch * len(train_loader) + i
            writer.add_scalar('ws_losses/condition_loss', condition_loss, step)
            writer.add_scalar('ws_losses/reconstruction_loss', reconstruction_loss, step)
            writer.add_scalar('ws_losses/diff_loss', diff_loss, step)
            writer.add_scalar('ws_losses/recognition_loss', sleep_loss.item(), step)
            writer.add_scalar('ws_losses/prior_loss', prior_loss.item(), step)
            writer.add_scalar('ws_losses/ce_loss', ce_loss, step)
            writer.add_scalar('ws_losses/entropy_loss', entropy_loss, step)
            writer.add_scalar('ws_losses/non_target_prob', non_target_prob, step)

            # 取最后一个batch的前5个进行plot
            if (i+1) == len(train_loader):
                x_inputs = [xs[j, :num_frames[j]].detach().cpu().numpy()
                            for j in range(cfg.TRAIN.PLOT_SAMPLES)]
                for j, (x, z_prob, z_pred, x_pred) in enumerate(zip(x_inputs, z_probs, z_preds, x_preds)):
                    writer.add_figure('tvs_compare/sample{}'.format(j),
                                      utils.plot_tvs([x, x_pred], z_pred),
                                      global_step=epoch)
                    temp = np.eye(cfg.MODEL.NUM_TARGETS)[z_pred].T  # [num_targets, num_frames]
                    insert_vector = np.array(
                        [[1.0 if temp[i].sum() > 0 else 0.0 for i in range(cfg.MODEL.NUM_TARGETS)]])  # [1, num_targets]
                    heatmap_array = np.concatenate([temp.T, insert_vector])  # [num_frames + 1, num_targets]
                    writer.add_figure('z_heatmap/sample{}'.format(j),
                                      utils.plot_heat_map(z_prob, heatmap_array),
                                      global_step=epoch)
                # if cfg.TRAIN.USE_STDS:
                #     means = generative_condition_model.target_means.detach().numpy()
                #     stds = generative_condition_model.target_stds.detach().numpy()
                #     writer.add_figure('targets_mean_std',
                #                       utils.plot_error_bar(means, stds),
                #                       global_step=epoch)

        if cfg.TRAIN.DO_EVAL:
            with torch.no_grad():
                inference_model.eval()
                generative_condition_model.eval()
                generative_prior_model.eval()

                eval_condition_loss_meter.reset()
                eval_reconstruction_loss_meter.reset()
                eval_diff_loss_meter.reset()
                eval_prior_loss_meter.reset()
                eval_recognition_loss_meter.reset()
                eval_ce_loss_meter.reset()
                eval_entropy_loss_meter.reset()

                pbar = tqdm(enumerate(eval_loader), total=len(eval_loader),
                            desc='Eval epoch {}/{}'.format(epoch, cfg.TRAIN.MAX_EPOCH))
                for i, (xs, num_frames) in pbar:
                    if cfg.TRAIN.DEVICE == 'gpu':
                        xs = xs.cuda()  # [batch_size, num_frame, 15]

                    condition_loss, reconstruction_loss, diff_loss, prior_loss, wake_interpolate_time, z_probs, z_preds, \
                        x_preds = losses.get_wake_loss(generative_prior_model, generative_condition_model, inference_model,
                                                       xs, num_frames, num_samples=cfg.TRAIN.NUM_SAMPLES,
                                                       plot_samples=cfg.TRAIN.PLOT_SAMPLES)
                    eval_condition_loss_meter.update(condition_loss)
                    eval_reconstruction_loss_meter.update(reconstruction_loss)
                    eval_diff_loss_meter.update(diff_loss)
                    eval_prior_loss_meter.update(prior_loss.item())

                    sleep_loss, sleep_interpolate_time, ce_loss, entropy_loss, non_target_prob = losses.get_sleep_loss(
                        generative_prior_model, generative_condition_model, inference_model,
                        cfg.TRAIN.NUM_SAMPLES * cfg.TRAIN.BATCH_SIZE)
                    eval_recognition_loss_meter.update(sleep_loss.item())
                    eval_ce_loss_meter.update(ce_loss)
                    eval_entropy_loss_meter.update(entropy_loss)

                    # 取最后一个batch的前5个进行plot
                    if (i + 1) == len(eval_loader):
                        x_inputs = [xs[j, :num_frames[j]].detach().cpu().numpy()
                                    for j in range(cfg.TRAIN.PLOT_SAMPLES)]
                        for j, (x, z_pred, x_pred) in enumerate(zip(x_inputs, z_preds, x_preds)):
                            writer.add_figure('eval_tvs_compare/sample{}'.format(j),
                                              utils.plot_tvs([x, x_pred], z_pred),
                                              global_step=epoch)

                writer.add_scalar('eval/condition_loss', eval_condition_loss_meter.avg, epoch)
                writer.add_scalar('eval/reconstruction_loss', eval_reconstruction_loss_meter.avg, epoch)
                writer.add_scalar('eval/diff_loss', eval_diff_loss_meter.avg, epoch)
                writer.add_scalar('eval/recognition_loss', eval_recognition_loss_meter.avg, epoch)
                writer.add_scalar('eval/prior_loss', eval_prior_loss_meter.avg, epoch)
                writer.add_scalar('eval/ce_loss', eval_ce_loss_meter.avg, epoch)
                writer.add_scalar('eval/entropy_loss', eval_entropy_loss_meter.avg, epoch)

                # pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
                # for i, (xs, num_frames) in pbar:
                #     if cfg.TRAIN.DEVICE == 'gpu':
                #         xs = xs.cuda()  # [batch_size, num_frame, 15]
                #
                #     condition_loss, prior_loss, wake_interpolate_time, z_probs, z_preds, x_preds = losses.get_wake_loss(
                #         generative_prior_model, generative_condition_model, inference_model,
                #         xs, num_frames, num_samples=cfg.TRAIN.NUM_SAMPLES,
                #         plot_samples=cfg.TRAIN.PLOT_SAMPLES)
                #
                #     zs = [z_prob.argmax(axis=-1) for z_prob in z_probs]
                #
                # xs = xs.cpu().numpy()
                # for i, (x, num_frame, z, x_pred) in enumerate(zip(xs[:cfg.TRAIN.PLOT_SAMPLES],
                #                                                   num_frames[:cfg.TRAIN.PLOT_SAMPLES], zs, x_preds)):
                #     writer.add_figure('tvs_inference_compare/sample{}'.format(i),
                #                       utils.plot_tvs(
                #                           [x[:num_frame], x_pred[:num_frame]], z[:num_frame]
                #                       ), global_step=epoch)

                # xs_ = xs_inference.to(device)
                # if cfg.TRAIN.AUTOREGRESSION:
                #     zs = inference_model.inference(xs_)
                # else:
                #     zs = (inference_model(xs_, num_frames_inference)).max(dim=-1)[1]
                # xs_ = xs_.cpu().numpy()
                #
                # x_preds, _ = generative_condition_model.sample_for_sleep(zs, num_frames_inference)
                # x_preds = x_preds.numpy()
                # zs = zs.cpu().numpy()
                # for i, (x, num_frame, z, x_pred) in enumerate(zip(xs_, num_frames_inference, zs, x_preds)):
                #     writer.add_figure('eval/tvs_compare/sample{}'.format(i),
                #                       utils.plot_tvs(
                #                           [x[:num_frame], x_pred[:num_frame]], z[:num_frame]
                #                       ), global_step=epoch)

            if epoch % cfg.TRAIN.SAVE_INTERVAL == 0:
                utils.save_training_checkpoint(inference_model, generative_prior_model, generative_condition_model,
                                               optimizer_inference,
                                               optimizer_generative_prior,
                                               optimizer_generative_condition,
                                               writer.log_dir,
                                               epoch,
                                               eval_recognition_loss_meter.avg,
                                               eval_reconstruction_loss_meter.avg,
                                               eval_prior_loss_meter.avg)


def train_prior(inference_model: models.InferenceModel, prior_model: models.GenerativeModelPriorDistribution,
                train_loader,
                prior_optimizer: torch.optim.Optimizer,
                writer: tensorboard.SummaryWriter,
                cfg):
    z_list = list()
    with torch.no_grad():
        inference_model.eval()
        for i, (xs, num_frames) in tqdm(enumerate(train_loader)):
            if cfg.TRAIN.DEVICE == 'gpu':
                xs = xs.cuda()
            zs = inference_model(xs, num_frames)
            for z, num_frame in zip(zs, num_frames):
                z = z[:num_frame].max(dim=-1)[1].cpu()
                z_list.append(z)

    zs_dataset = PriorDataset(z_list)
    zs_loader = DataLoader(zs_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True,
                           num_workers=cfg.TRAIN.NUM_LOADER_WORKERS,
                           collate_fn=collate_tvs,
                           pin_memory=True, drop_last=True)

    prior_model.train()
    for epoch in range(cfg.TRAIN.PRIOR_EPOCHS):
        pbar = tqdm(enumerate(zs_loader), desc='Epoch {}'.format(epoch), total=len(zs_loader))
        for i, (zs, num_frames) in pbar:
            zs = zs.cuda()
            prior_optimizer.zero_grad()

            prior_loss = prior_model.log_prob(zs, num_frames).mean()

            prior_loss.backward()

            prior_optimizer.step()

            writer.add_scalar('prior/loss', prior_loss.item(), global_step=epoch*len(zs_loader)+i)

            pbar.set_description("Epoch_iter {}_{}: "
                                 "loss: {:.2f}".format(epoch, i, prior_loss.item()))





