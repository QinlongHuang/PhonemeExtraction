# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/run.py
# @Author: Qinlong Huang
# @Create Date: 2021/04/09 13:20
# @Contact: qinlonghuang@gmail.com
# @Description:

import sys
sys.path.append('..')

import random

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from functools import reduce
from torch.utils.tensorboard import SummaryWriter
import utils

import ray

from src.train import train_ws, train_sleep, train_prior
from src.config import cfg
from src.data import TvsDataset, EMADataset, M01_Dataset, collate_tvs
from src.models import GenerativeModelPriorDistribution, GenerativeModelConditionDistribution, InferenceModel

WORKER_SEED = 1

torch.autograd.set_detect_anomaly(True)


def worker_init(worker_id):
    np.random.seed(int(WORKER_SEED + worker_id))


def print_model_info(model, model_name):
    nparams = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            p_shape = list(param.shape)
            p_n = reduce(lambda x, y: x * y, p_shape)
            nparams += p_n
            print('{} ({}): {}, device: {}'.format(p_shape, p_n, name, param.device))
    print('{} Total params: {} ({:.2f} MB)'.format(model_name, nparams, (float(nparams) * 4) / (1024 * 1024)))
    print('-' * 80)


def run():

    ray.init(num_cpus=cfg.TRAIN.NUM_PROCESSES, num_gpus=0)
    # --------------------
    #     Data Process
    # --------------------
    if not cfg.TRAIN.EMA:
        train_dataset = TvsDataset(cfg.TRAIN.TRAINING_DATA_PATH, cfg.TRAIN.TRAINING_SAMPLES)
        eval_dataset = TvsDataset(cfg.TRAIN.EVAL_DATA_PATH, cfg.TRAIN.TRAINING_SAMPLES)
    else:
        train_dataset = EMADataset(cfg.TRAIN.SPEAKER_NAME)
        eval_dataset = EMADataset(cfg.TRAIN.SPEAKER_NAME)
    train_loader = DataLoader(train_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True,
                            num_workers=cfg.TRAIN.NUM_LOADER_WORKERS,
                            collate_fn=collate_tvs,
                            pin_memory=True, drop_last=True)

    eval_loader = DataLoader(eval_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.NUM_LOADER_WORKERS,
                            collate_fn=collate_tvs,
                            pin_memory=True, drop_last=True)

    # ---------------------------------
    #     Build Model and Optimizer
    # ---------------------------------
    print('Building model...')
    generative_prior_model = GenerativeModelPriorDistribution(lstm_input_dim=cfg.MODEL.NUM_TARGETS,
                                                            lstm_hidden_dim=cfg.MODEL.GENERATIVE_LSTM_HIDDEN_DIM,
                                                            num_layers=cfg.MODEL.PRIOR_NUM_LAYERS,
                                                            num_targets=cfg.MODEL.NUM_TARGETS)
    generative_condition_model = GenerativeModelConditionDistribution(num_targets=cfg.MODEL.NUM_TARGETS,
                                                                    tvs_dim=cfg.MODEL.TVS_DIM,
                                                                    use_stds=cfg.TRAIN.USE_STDS,
                                                                    num_processes=cfg.TRAIN.NUM_PROCESSES,
                                                                    mode=cfg.MODEL.INTERPOLATION_MODE,
                                                                    diff_lambda=cfg.MODEL.RECONSTRUCTION_DIFF_LAMBDA,
                                                                    diff_n=cfg.MODEL.RECONSTRUCTION_DIFF_N)
    generative_condition_model.share_memory()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    if cfg.TRAIN.TRM_INIT:
        generative_condition_model.target_means = nn.Parameter(
            torch.tensor([
                [0, 0, 0, 0, 5.5, 2500, 500, 0.89, 0.99, 0.81, 0.76, 1.05, 1.23, 0.01, 0.1],  # silence
                [0, 60, 0, 0, 5.5, 2500, 500, 0.65, 0.65, 0.65, 1.31, 1.23, 1.31, 1.67, 0.1],  # a
                [0, 60, 0, 0, 5.5, 2500, 500, 0.65, 0.84, 1.15, 1.31, 1.59, 1.59, 2.61, 0.1],  # aa
                [0, 60, 0, 0, 5.5, 2500, 500, 0.65, 0.45, 0.94, 1.10, 1.52, 1.46, 2.45, 0.1],  # ah
                [0, 60, 0, 0, 5.5, 2500, 500, 0.52, 0.45, 0.79, 1.49, 1.67, 1.02, 1.59, 1.5],  # an
                [0, 60, 0, 0, 5.5, 2500, 500, 0.52, 0.45, 0.79, 1.49, 1.67, 1.02, 1.59, 0.1],  # ar
                [0, 60, 0, 0, 5.5, 2500, 500, 1.10, 0.94, 0.42, 1.49, 1.67, 1.78, 1.05, 0.1],  # aw
                [-2, 43.5, 0, 0, 7, 2000, 700, 0.89, 0.76, 1.28, 1.80, 0.99, 0.84, 0.10, 0.1],  # b
                [-2, 0, 0, 0, 5.6, 2500, 2600, 1.36, 1.74, 1.87, 0.94, 0.00, 0.79, 0.79, 0.1],  # ch
                [-2, 43.5, 0, 0, 6.7, 4500, 2000, 1.31, 1.49, 1.25, 0.76, 0.10, 1.44, 1.30, 0.1],  # d
                [-1, 54, 0, 0.25, 6, 4400, 4500, 1.20, 1.50, 1.35, 1.20, 1.20, 0.40, 1.00, 0.1],  # dh
                [-2, 43.5, 0, 0, 6.7, 4500, 2000, 1.31, 1.49, 1.25, 0.76, 0.10, 1.44, 1.31, 0.1],  # dx
                [0, 60, 0, 0, 5.5, 2500, 500, 0.68, 1.12, 1.695, 1.385, 1.07, 1.045, 2.06, 0.1],  # e
                [0, 60, 0, 0, 5.5, 2500, 500, 1.67, 1.905, 1.985, 0.81, 0.495, 0.73, 1.485, 0.1],  # ee
                [0, 60, 0, 0, 5.5, 2500, 500, 0.885, 0.99, 0.81, 0.755, 1.045, 1.225, 1.12, 0.1],  # er
                [-1, 0, 0, 0.5, 7, 3300, 1000, 0.89, 0.99, 0.81, 0.76, 0.89, 0.84, 0.50, 0.1],  # f
                [-2, 43.5, 0, 0, 4.7, 2000, 2000, 1.70, 1.30, 0.99, 0.10, 1.07, 0.73, 1.49, 0.1],  # g
                [0, 0, 0, 0, 5.5, 2500, 500, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.1],  # gs
                [0, 0, 10, 0, 5.5, 2500, 500, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.1],  # h
                [0, 0, 10, 0, 1, 1000, 1000, 0.24, 0.40, 0.81, 0.76, 1.05, 1.23, 1.12, 0.1],  # hh
                [0, 42, 10, 0, 5.5, 2500, 500, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.1],  # hv
                [0, 60, 0, 0, 5.5, 2500, 500, 1.045, 1.565, 1.75, 0.94, 0.68, 0.785, 1.12, 0.1],  # i
                [0, 60, 0, 0, 5.5, 2500, 500, 0.65, 0.835, 1.15, 1.305, 1.59, 1.59, 2.61, 1.5],  # in
                [-2, 48, 0, 0, 5.6, 2500, 2600, 1.36, 1.74, 1.87, 0.94, 0.00, 0.79, 0.79, 0.1],  # j
                [-10, 0, 0, 0, 4.7, 2000, 2000, 1.70, 1.30, 0.99, 0.10, 1.07, 0.73, 1.49, 0.1],  # k
                [0, 60, 0, 0, 5.5, 2500, 500, 0.89, 1.10, 0.97, 0.89, 0.34, 0.29, 1.12, 0.1],  # l
                [0, 60, 0, 0, 5.5, 2500, 500, 0.63, 0.47, 0.65, 1.54, 0.45, 0.26, 1.05, 0.1],  # ll
                [0, 60, 0, 0, 5.5, 2500, 500, 0.89, 0.76, 1.28, 1.80, 0.99, 0.84, 0.10, 0.5],  # m
                [0, 60, 0, 0, 5.5, 2500, 500, 1.31, 1.49, 1.25, 1.00, 0.05, 1.44, 1.31, 0.5],  # n
                [0, 60, 0, 0, 5.5, 2500, 500, 1.70, 1.30, 0.99, 0.10, 1.07, 0.73, 1.49, 0.5],  # ng
                [0, 60, 0, 0, 5.5, 2500, 500, 1.00, 0.925, 0.60, 1.27, 1.83, 1.97, 1.12, 0.1],  # o
                [0, 0, 0, 0, 5.5, 2500, 500, 0.885, 0.99, 0.81, 0.755, 1.045, 1.225, 1.12, 0.1],  # oh
                [0, 60, 0, 0, 5.5, 2500, 500, 1.00, 0.925, 0.60, 1.265, 1.83, 1.965, 1.12, 1.5],  # on
                [-10, 0, 0, 0, 7, 2000, 700, 0.89, 0.76, 1.28, 1.80, 0.99, 0.84, 0.10, 0.1],  # p
                [-1, 0, 0, 24, 7, 864, 3587, 0.89, 0.99, 0.81, 0.60, 0.52, 0.71, 0.24, 0.1],  # ph
                [-2, 0, 0, 0, 5.6, 2500, 2600, 1.36, 1.74, 1.87, 0.94, 0.10, 0.79, 0.79, 0.1],  # qc
                [-10, 0, 0, 0, 4.7, 2000, 2000, 1.70, 1.30, 0.99, 0.10, 1.07, 0.73, 1.49, 0.1],  # qk
                [0, 0, 0, 0, 5.8, 5500, 500, 1.31, 1.49, 1.25, 0.90, 0.20, 0.40, 1.31, 0.1],  # qs
                [-1, 0, 0, 0, 5.8, 5500, 500, 1.31, 1.49, 1.25, 0.99, 0.20, 0.60, 1.31, 0.1],  # qz
                [0, 60, 0, 0, 5.5, 2500, 500, 1.31, 0.73, 1.07, 2.12, 0.47, 1.78, 0.65, 0.1],  # r
                [0, 60, 0, 0, 5.5, 2500, 500, 1.31, 0.73, 1.31, 2.12, 0.63, 1.78, 0.65, 0.1],  # rr
                [0, 0, 0, 0.8, 5.8, 5500, 500, 1.31, 1.49, 1.25, 0.90, 0.20, 0.40, 1.31, 0.1],  # s
                [0, 0, 0, 0.4, 5.6, 2500, 2600, 1.36, 1.74, 1.87, 0.94, 0.37, 0.79, 0.79, 0.1],  # sh
                [-10, 0, 0, 0, 7, 4500, 2000, 1.31, 1.49, 1.25, 0.76, 0.10, 1.44, 1.31, 0.1],  # t
                [0, 0, 0, 0.25, 6, 4400, 4500, 1.20, 1.50, 1.35, 1.20, 1.20, 0.40, 1.00, 0.1],  # th
                [-10, 0, 0, 0, 6.7, 4500, 2000, 1.31, 1.49, 1.25, 0.76, 0.10, 1.44, 1.31, 0.1],  # tx
                [0, 60, 0, 0, 5.5, 2500, 500, 0.625, 0.60, 0.705, 1.12, 1.93, 1.515, 0.625, 0.1],  # u
                [0, 60, 0, 0, 5.5, 2500, 500, 0.89, 0.99, 0.81, 0.76, 1.05, 1.23, 1.21, 0.1],  # uh
                [0, 60, 0, 0, 5.5, 2500, 500, 0.885, 0.99, 0.81, 0.755, 1.045, 1.225, 1.12, 1.5],  # un
                [-1, 54, 0, 0.2, 7, 3300, 1000, 0.89, 0.99, 0.81, 0.76, 0.89, 0.84, 0.50, 0.1],  # v
                [0, 60, 0, 0, 5.5, 2500, 500, 1.91, 1.44, 0.60, 1.02, 1.33, 1.56, 0.55, 0.1],  # w
                [0, 0, 0, 0.5, 2, 1770, 900, 1.70, 1.30, 0.40, 0.99, 1.07, 0.73, 1.49, 0.1],  # x
                [0, 60, 0, 0, 5.5, 2500, 500, 1.67, 1.91, 1.99, 0.63, 0.29, 0.58, 1.49, 0.25],  # y
                [-1, 54, 0, 0.8, 5.8, 5500, 500, 1.31, 1.49, 1.25, 0.90, 0.20, 0.60, 1.31, 0.1],  # z
                [-1, 54, 0, 0.4, 5.6, 2500, 2600, 1.36, 1.74, 1.87, 0.94, 0.37, 0.79, 0.79, 0.1]  # zh
            ])
        )
    inference_model = InferenceModel(lstm_hidden_dim=cfg.MODEL.INFERENCE_LSTM_HIDDEN_DIM,
                                     obs_embedding_dim=cfg.MODEL.TVS_DIM,
                                     num_layers=cfg.MODEL.INFERENCE_NUM_LAYERS,
                                     num_targets=cfg.MODEL.NUM_TARGETS,
                                     entropy_lambda=cfg.MODEL.INFERENCE_ENTROPY_LAMBDA,
                                     non_target_lambda=cfg.MODEL.NON_TARGET_LAMBDA,
                                     auto_regression=cfg.TRAIN.AUTOREGRESSION)

    # TODO: restore model parameters
    if cfg.TRAIN.DEVICE == 'gpu':
        print('{} GPUs specified to use, {} GPUs available.'.format(len(cfg.TRAIN.GPU_IDS), torch.cuda.device_count()))
        print('torch version: {}, cuda version: {}, cudnn version: {}'.format(torch.__version__, torch.version.cuda,
                                                                              torch.backends.cudnn.version()))
        if len(cfg.TRAIN.GPU_IDS) > 1:
            generative_prior_model = nn.DataParallel(generative_prior_model, device_ids=cfg.TRAIN.GPU_IDS)
            inference_model = nn.DataParallel(inference_model, device_ids=cfg.TRAIN.GPU_IDS)
        else:
            generative_prior_model.cuda()
            inference_model.cuda()
    print_model_info(generative_prior_model, 'Generative Prior Model')
    print_model_info(generative_condition_model, 'Generative Condition Model')
    print_model_info(inference_model, 'Inference Model')
    print('Finish !')

    print('Building optimizer...')
    assert cfg.TRAIN.OPTIMIZER in ['Adam', 'AdamW', 'SGD']
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer_generative_prior = optim.Adam(generative_prior_model.parameters(), lr=cfg.TRAIN.LR)
        optimizer_generative_condition = optim.Adam(generative_condition_model.parameters(), lr=cfg.TRAIN.LR)
        optimizer_inference = optim.Adam(inference_model.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'AdamW':
        optimizer_generative_prior = optim.AdamW(generative_prior_model.parameters(), lr=cfg.TRAIN.LR)
        optimizer_generative_condition = optim.AdamW(generative_condition_model.parameters(), lr=cfg.TRAIN.LR)
        optimizer_inference = optim.AdamW(inference_model.parameters(), lr=cfg.TRAIN.LR)
    else:
        optimizer_generative_prior = optim.SGD(generative_prior_model.parameters(), lr=cfg.TRAIN.LR)
        optimizer_generative_condition = optim.SGD(generative_condition_model.parameters(), lr=cfg.TRAIN.LR)
        optimizer_inference = optim.SGD(inference_model.parameters(), lr=cfg.TRAIN.LR)
    print('Finish !')

    # ----------------
    #     Training
    # ----------------
    log_dir = cfg.TRAIN.WS_MODEL_PATH if cfg.TRAIN.RESTORE and cfg.TRAIN.INPLACE \
        else os.path.join(cfg.TRAIN.LOG_DIR, cfg.TRAIN.EXP_NAME)
    writer = SummaryWriter(log_dir=log_dir)
    # pretrain in sleep phase
    if cfg.TRAIN.RESTORE:
        if cfg.TRAIN.WS_MODEL_PKL:
            checkpoint = torch.load(os.path.join(cfg.TRAIN.WS_MODEL_PATH, cfg.TRAIN.WS_MODEL_PKL))

            inference_model.load_state_dict(checkpoint['inference_model_state_dict'])
            generative_prior_model.load_state_dict(checkpoint['generative_prior_model_state_dict'])
            generative_condition_model.load_state_dict(checkpoint['generative_condition_model_state_dict'])
            optimizer_inference.load_state_dict(checkpoint['optimizer_inference_state_dict'])
            optimizer_generative_prior.load_state_dict(checkpoint['optimizer_generative_prior_state_dict'])
            optimizer_generative_condition.load_state_dict(checkpoint['optimizer_generative_condition_state_dict'])

            print('Loaded pretrained WS model in {} !'.format(cfg.TRAIN.WS_MODEL_PATH))
        elif cfg.TRAIN.PRETRAINED_SLEEP_MODEL_PATH:
            load_path = os.path.join(log_dir, cfg.TRAIN.PRETRAINED_SLEEP_MODEL_PATH)
            if cfg.TRAIN.DEVICE == 'gpu':
                checkpoints = torch.load(load_path)
            else:
                checkpoints = torch.load(load_path,
                                         map_location=lambda storage, loc: storage)

            inference_model.load_state_dict(checkpoints['model_state_dict'])

            print('Loaded pretrainded sleep model in {} !'.format(cfg.TRAIN.PRETRAINED_SLEEP_MODEL_PATH))
        else:
            raise RuntimeError('Please assign the path of the model you want to restore !')
    else:
        sleep_loss = train_sleep(generative_prior_model,
                                 generative_condition_model,
                                 inference_model,
                                 cfg.TRAIN.LR * cfg.TRAIN.PRIOR_LR_FACTOR,
                                 cfg.TRAIN.NUM_SAMPLES * cfg.TRAIN.BATCH_SIZE,
                                 cfg.TRAIN.PRETRAIN_EPOCH,
                                 writer)
        utils.save_pretrain_checkpoint(inference_model, optimizer_inference, log_dir, sleep_loss)

    # training
    train_ws(generative_prior_model,
             generative_condition_model,
             inference_model,
             train_loader,
             eval_loader,
             optimizer_generative_prior, optimizer_generative_condition, optimizer_inference,
             writer,
             0 if not cfg.TRAIN.RESTORE else cfg.TRAIN.RESTORE_EPOCHS,
             cfg)

    ray.shutdown()


def run_prior_training():

    train_dataset = TvsDataset(cfg.TRAIN.TRAINING_DATA_PATH, cfg.TRAIN.TRAINING_SAMPLES)
    train_loader = DataLoader(train_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.NUM_LOADER_WORKERS,
                              collate_fn=collate_tvs,
                              pin_memory=True, drop_last=True)

    generative_prior_model = GenerativeModelPriorDistribution(lstm_input_dim=cfg.MODEL.NUM_TARGETS,
                                                              lstm_hidden_dim=cfg.MODEL.GENERATIVE_LSTM_HIDDEN_DIM,
                                                              num_layers=cfg.MODEL.PRIOR_NUM_LAYERS,
                                                              num_targets=cfg.MODEL.NUM_TARGETS)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')

    inference_model = InferenceModel(lstm_hidden_dim=cfg.MODEL.INFERENCE_LSTM_HIDDEN_DIM,
                                     obs_embedding_dim=cfg.MODEL.TVS_DIM,
                                     num_layers=cfg.MODEL.INFERENCE_NUM_LAYERS,
                                     num_targets=cfg.MODEL.NUM_TARGETS,
                                     entropy_lambda=cfg.MODEL.INFERENCE_ENTROPY_LAMBDA,
                                     non_target_lambda=cfg.MODEL.NON_TARGET_LAMBDA,
                                     auto_regression=cfg.TRAIN.AUTOREGRESSION)

    if cfg.TRAIN.DEVICE == 'gpu':
        print('{} GPUs specified to use, {} GPUs available.'.format(len(cfg.TRAIN.GPU_IDS), torch.cuda.device_count()))
        print('torch version: {}, cuda version: {}, cudnn version: {}'.format(torch.__version__, torch.version.cuda,
                                                                              torch.backends.cudnn.version()))
        if len(cfg.TRAIN.GPU_IDS) > 1:
            generative_prior_model = nn.DataParallel(generative_prior_model, device_ids=cfg.TRAIN.GPU_IDS)
            inference_model = nn.DataParallel(inference_model, device_ids=cfg.TRAIN.GPU_IDS)
        else:
            generative_prior_model.cuda()
            inference_model.cuda()
    print_model_info(generative_prior_model, 'Generative Prior Model')
    print_model_info(inference_model, 'Inference Model')
    print('Finish !')

    optimizer_generative_prior = optim.Adam(generative_prior_model.parameters(), lr=cfg.TRAIN.LR)

    log_dir = cfg.TRAIN.WS_MODEL_PATH if cfg.TRAIN.RESTORE and cfg.TRAIN.INPLACE \
        else os.path.join(cfg.TRAIN.LOG_DIR, cfg.TRAIN.EXP_NAME)
    writer = SummaryWriter(log_dir=log_dir)

    if cfg.TRAIN.WS_MODEL_PKL:
        checkpoint = torch.load(os.path.join(cfg.TRAIN.WS_MODEL_PATH, cfg.TRAIN.WS_MODEL_PKL))
        inference_model.load_state_dict(checkpoint['inference_model_state_dict'])

        print('Loaded pretrained WS model in {} !'.format(cfg.TRAIN.WS_MODEL_PATH))

    train_prior(inference_model, generative_prior_model, train_loader, optimizer_generative_prior,
                writer, cfg)


if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(','.join([str(idx) for idx in cfg.TRAIN.GPU_IDS]))
    # 使用ray需要加这一段来设置环境变量
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
    utils.setup_seed(666)

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (51200, rlimit[1]))

    with torch.autograd.set_detect_anomaly(True):
        while True:
            try:
                # run_prior_training()
                run()
                break
            except RuntimeError as e:
                if "CUDA out of memory" in repr(e):
                    cfg.TRAIN.BATCH_SIZE -= 1
                    print(e)
                    print("Decreasing batch_size to: {}".format(cfg.TRAIN.BATCH_SIZE))
                    if cfg.TRAIN.BATCH_SIZE == 0:
                        raise RuntimeError("batch size got decreased to 0")
                    else:
                        continue
                else:
                    raise e
