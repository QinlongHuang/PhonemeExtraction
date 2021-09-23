# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/config.py
# @Author: Qinlong Huang
# @Create Date: 2021/04/09 15:47
# @Contact: qinlonghuang@gmail.com
# @Description:

import numpy as np
from easydict import EasyDict as edict
from datetime import datetime
from pathlib import Path

data_root = Path('/data1/huangqinlong/PhonemeExtraction/data')

__C = edict()
cfg = __C

__C.DATA = edict()
__C.DATA.speaker_arti_csv = data_root / 'articulators_per_speaker.csv'
__C.DATA.mocha_root = data_root / 'mocha-timit'

__C.MODEL = edict()
__C.MODEL.GENERATIVE_LSTM_HIDDEN_DIM = 512  # 128, 512
__C.MODEL.INFERENCE_LSTM_HIDDEN_DIM = 256  # 64, 256, 512
__C.MODEL.INFERENCE_NUM_LAYERS = 2
__C.MODEL.PRIOR_NUM_LAYERS = 1
__C.MODEL.TVS_DIM = 15  # 12, 15
__C.MODEL.NUM_TARGETS = 64  # 64, 46 means number of phonemes in mocha-timit
__C.MODEL.INFERENCE_ENTROPY_LAMBDA = 0.1
__C.MODEL.INTERPOLATION_MODE = 'pchip'  # linear, pchip
__C.MODEL.RECONSTRUCTION_DIFF_LAMBDA = 0.0
__C.MODEL.RECONSTRUCTION_DIFF_N = 1
__C.MODEL.NON_TARGET_LAMBDA = 0.0

__C.TRAIN = edict()
__C.TRAIN.EMA = True
__C.TRAIN.SPEAKER_NAME = 'msak0'
__C.TRAIN.TRAINING_DATA_PATH = '/data1/huangqinlong/PhonemeExtraction/data/train'
__C.TRAIN.EVAL_DATA_PATH = '/data1/huangqinlong/PhonemeExtraction/data/eval'
__C.TRAIN.DO_EVAL = False
__C.TRAIN.DEVICE = 'gpu'
__C.TRAIN.GPU_IDS = [2]
__C.TRAIN.USE_STDS = True
__C.TRAIN.NUM_SAMPLES = 1
__C.TRAIN.SAVE_INTERVAL = 10
__C.TRAIN.NUM_LOADER_WORKERS = 0
__C.TRAIN.BATCH_SIZE = 48  # 96
__C.TRAIN.MAX_EPOCH = 1000
__C.TRAIN.ALGORITHM = "ws"
__C.TRAIN.LR = 1e-3
__C.TRAIN.PRIOR_LR_FACTOR = 2.0
__C.TRAIN.PRETRAIN_EPOCH = 100
__C.TRAIN.OPTIMIZER = 'AdamW'
__C.TRAIN.NUM_PROCESSES = 64
__C.TRAIN.TRM_INIT = False
__C.TRAIN.AUTOREGRESSION = False
__C.TRAIN.TRAINING_SAMPLES = 5400  # 5400, 540 for tvs, 460 for mocha-timit, 800 for ema
__C.TRAIN.PRIOR_EPOCHS = 500

__C.TRAIN.LOG_DIR = '/data1/huangqinlong/PhonemeExtraction/log_dir'
__C.TRAIN.RESTORE = False
__C.TRAIN.INPLACE = False
__C.TRAIN.PRETRAINED_SLEEP_MODEL_PATH = 'PRETRAINED_INFERENCE_MODEL_loss=-9.2528.pkl'
__C.TRAIN.WS_MODEL_PATH = '/data1/huangqinlong/PhonemeExtraction/previous/tmp/' \
                          'LinearInterpolation-Use_Stds-lr=0.001-entropy_lambda=0.1-DynamicRNN-targets=500-interpolation=pchip-Large-model-CrossEntropy-Tvs-num=15-Low-pass-20210513-22:03:13'
__C.TRAIN.WS_MODEL_PKL = 'WS_MODEL_Epoch430_recognition_loss=0.0188_condition_loss=0.1330_prior_loss=1.5067.pkl'
__C.TRAIN.RESTORE_EPOCHS = 111
__C.TRAIN.PLOT_SAMPLES = 5
__C.TRAIN.EXP_NAME = '{}-' \
                     '{}-' \
                     'lr={}-' \
                     'entropy_lambda={}-' \
                     '{}-' \
                     'targets={}-' \
                     'interpolation={}-' \
                     'hidden={}+{}+{}-' \
                     'CrossEntropy-' \
                     'Tvs-num={}-' \
                     'diff={}-{}-' \
                     'ntl={}-' \
                     '{}'.format(__C.TRAIN.TRAINING_DATA_PATH.split('/')[-1],
                                 'Use_Stds' if __C.TRAIN.USE_STDS else 'Unused_stds',
                                 __C.TRAIN.LR,
                                 __C.MODEL.INFERENCE_ENTROPY_LAMBDA,
                                 'AutoRegression' if __C.TRAIN.AUTOREGRESSION else 'DynamicRNN',
                                 __C.MODEL.NUM_TARGETS,
                                 __C.MODEL.INTERPOLATION_MODE,
                                 __C.MODEL.GENERATIVE_LSTM_HIDDEN_DIM,
                                 __C.MODEL.INFERENCE_LSTM_HIDDEN_DIM,
                                 __C.MODEL.INFERENCE_NUM_LAYERS,
                                 __C.MODEL.TVS_DIM,
                                 __C.MODEL.RECONSTRUCTION_DIFF_LAMBDA,
                                 __C.MODEL.RECONSTRUCTION_DIFF_N,
                                 __C.MODEL.NON_TARGET_LAMBDA,
                                 datetime.now().strftime('%Y%m%d-%H:%M:%S'))

