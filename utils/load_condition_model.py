# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/load_condition_model.py
# @Author: Qinlong Huang
# @Create Date: 2021/05/13 14:58
# @Contact: qinlonghuang@gmail.com
# @Description:

import torch
from torch import nn
from glob import glob
import numpy as np
from tqdm import tqdm


class GenerativeModelConditionDistribution(nn.Module):

    def __init__(self, num_targets, tvs_dim, use_stds, num_processes=32, mode='linear'):
        """
        :param num_targets: int
        :param tvs_dim: int
        :param num_processes: int
        """
        super().__init__()

        self.num_targets = num_targets
        self.num_processes = num_processes
        self.use_stds = use_stds
        self.target_means = nn.Parameter(torch.randn(num_targets, tvs_dim).clamp(-1, 1))
        self.target_stds = nn.Parameter(torch.randn(num_targets, tvs_dim).clamp(-1, 1)) if use_stds else None
        self.mode = mode


# Usage
import os
model_dir = "/data1/huangqinlong/PhonemeExtraction/model_files"
model_fps = glob(os.path.join(model_dir, '*.pkl'))
target_dir = "/data1/huangqinlong/PhonemeExtraction/target_files"

# pbar = tqdm(model_fps)
# for fp in pbar:
#     file_name = fp.split('/')[-1].split('.')[0]  # e.g., targets=500_interpolation=pchip_hidden=512
#     targets = int(file_name.split('_')[0].split('=')[-1])  # 500
#     generative_condition_model = GenerativeModelConditionDistribution(num_targets=targets,  # 128, 500
#                                                                       tvs_dim=15,
#                                                                       use_stds=True,
#                                                                       num_processes=64,
#                                                                       mode='pchip')  # 'pchip'
#     checkpoint = torch.load(fp)
#     generative_condition_model.load_state_dict(checkpoint['generative_condition_model_state_dict'])
#     means = generative_condition_model.target_means.detach().numpy()
#     new_fp = os.path.join(target_dir, file_name+'.npy')
#     np.save(new_fp, means)


import glob
import pickle

frame_len = 100
npy_fps = glob.glob(os.path.join(target_dir, '*.npy'))
wav_root = '/data1/huangqinlong/PhonemeExtraction/target_wavs'
track = np.array([-0.9880, -0.0500, -1.0000, -0.9987, -0.9976, -0.9992, -1.0000, -0.6375,
                  -0.7450, 0.9992, 0.9994, -1.0000, 0.9996])
for npy_fp in npy_fps:
    targets = np.load(npy_fp)
    wav_dir_fp = npy_fp.split('/')[-1].split('.')[0]
    wav_path = os.path.join(wav_root, wav_dir_fp)
    if not os.path.exists(wav_path):
        os.mkdir(wav_path)

    track_tvs_fp_list = list()
    save_fp = 'track_tvs_fps.pkl'
    for i, target in enumerate(targets):
        tvs = np.stack([target]*frame_len, axis=0)
        r0_vec = np.zeros((tvs.shape[0], 1))
        tvs = np.concatenate((tvs[:, :7], r0_vec, tvs[:, 7:]), axis=-1)
        fp = os.path.join(wav_path, '{}_trm_params.txt'.format(i))
        track_tvs_fp_list.append((track, tvs, fp))

    fp = os.path.join(wav_path, save_fp)
    with open(fp, 'wb') as fw:
        pickle.dump(track_tvs_fp_list, fw)
        print('Saved track_tvs_fps into {}'.format(fp))








