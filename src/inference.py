# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/inference.py
# @Author: Qinlong Huang
# @Create Date: 2021/05/28 14:31
# @Contact: qinlonghuang@gmail.com
# @Description:

import os
import re
import errno
import torch
from tqdm import tqdm
from src.models import InferenceModel, GenerativeModelConditionDistribution
import numpy as np
import glob
from torch.utils.data import DataLoader
from src.data import TvsDataset, TrackTvsDataset, collate_tvs, collate_track_tvs
import pickle
from src.config import cfg
import ray
import matplotlib.pyplot as plt
from src.plot_curves import plot_single_ori_rec_tvs
from src import utils


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_filename(fp):
    regex = "([^/]+)$"
    matchObj = re.search(regex, fp)
    if matchObj:
        filename = matchObj.group()
    else:
        raise ValueError("No match filename in {}".format(fp))
    return filename


def extract_tvs_for_generating(inference_model: InferenceModel,
                               generative_condition_model: GenerativeModelConditionDistribution,
                               ref_path, source_dir, target_dir):

    with torch.no_grad():
        inference_model.eval()
        generative_condition_model.eval()
        x_inference_list = list()
        track_list = list()

        ref_fps = glob.glob(os.path.join(ref_path, '*.pkl'))
        reconstructed_trm_params_fps = []
        for fp in ref_fps:
            trm_fp = fp.replace(source_dir.strip('/'), target_dir.strip('/')).replace('_spec_tvs_track.pkl',
                                                                                      '_trm_params.txt')
            mkdir_p(trm_fp.strip(get_filename(trm_fp)))  # recursively make dir
            reconstructed_trm_params_fps.append(trm_fp)

        dataset = TrackTvsDataset(ref_fps)
        eval_loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE * 2, shuffle=False,
                                 collate_fn=collate_track_tvs, pin_memory=True)

        pbar = tqdm(eval_loader, total=len(eval_loader), desc='Evaluating...')
        for tracks, xs, num_frames in pbar:
            if cfg.TRAIN.DEVICE == 'gpu':
                xs = xs.cuda()  # [batch_size, num_frame, 15]

            z_logits = inference_model(xs, num_frames).cpu()
            z_inference_batch = z_logits.argmax(dim=-1)
            x_inference_batch, sleep_interpolate_time = generative_condition_model.sample_for_sleep(z_inference_batch,
                                                                                                    num_frames)
            for i, (x_inference, track) in enumerate(zip(x_inference_batch, tracks)):
                x_inference = x_inference[:num_frames[i]].numpy()
                r0_vec = np.zeros((x_inference.shape[0], 1))
                x_inference = np.concatenate((x_inference[:, :7], r0_vec, x_inference[:, 7:]), axis=-1)
                x_inference_list.append(x_inference)
                track_list.append(track)

        track_tvs_fp_list = list()
        for track, tvs, fp in zip(track_list, x_inference_list, reconstructed_trm_params_fps):
            track_tvs_fp_list.append((track, tvs, fp))

        fp = os.path.join(ref_path, '..', '20210424-00:43:21_track_tvs_fps.pkl')
        with open(fp, 'wb') as fw:
            pickle.dump(track_tvs_fp_list, fw)
            print('Saved track_tvs_fps into {}'.format(fp))


def inference(inference_model: InferenceModel,
              generative_condition_model: GenerativeModelConditionDistribution,
              data_path,
              write_fp,
              mode='argmax'):
    assert mode in ['argmax', 'sampling']
    with torch.no_grad():
        inference_model.eval()
        generative_condition_model.eval()
        x_ori_inference_list = list()
        z_logits_x_ori_inference_list = list()

        dataset = TvsDataset(data_path)
        eval_loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE * 2, shuffle=False,
                                 collate_fn=collate_tvs, pin_memory=True)
        pbar = tqdm(eval_loader, total=len(eval_loader), desc='Evaluating...')
        for xs, num_frames in pbar:
            if cfg.TRAIN.DEVICE == 'gpu':
                xs = xs.cuda()  # [batch_size, num_frame, 15]

            z_logits = inference_model(xs, num_frames).cpu()  # [batch_size, num_frame, num_targets]
            if mode == 'argmax':
                z_inference_batch = z_logits.argmax(dim=-1)
            else:
                z_inference_batch = torch.distributions.Categorical(logits=z_logits).sample()
            x_inference_batch, sleep_interpolate_time = generative_condition_model.sample_for_sleep(z_inference_batch,
                                                                                                    num_frames)

            for i, (x_ori, x_inference, z_logit) in enumerate(zip(xs, x_inference_batch, z_logits)):
                x_ori = x_ori.cpu().numpy()[:num_frames[i]]
                x_inference = x_inference[:num_frames[i]].numpy()
                # x_ori_inference_list.append(np.stack([x_ori, x_inference], axis=0))
                z_logits_x_ori_inference_list.append((z_logit[:num_frames[i]].numpy(),
                                                      np.stack([x_ori, x_inference], axis=0)))

        with open(write_fp, 'wb') as fw:
            pickle.dump(z_logits_x_ori_inference_list, fw)

        return x_ori_inference_list


if __name__ == '__main__':

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(','.join([str(idx) for idx in cfg.TRAIN.GPU_IDS]))

    ray.init(num_cpus=64, num_gpus=0)
    inference_model_ = InferenceModel(lstm_hidden_dim=cfg.MODEL.INFERENCE_LSTM_HIDDEN_DIM,
                                      obs_embedding_dim=cfg.MODEL.TVS_DIM,
                                      num_layers=cfg.MODEL.INFERENCE_NUM_LAYERS,
                                      num_targets=cfg.MODEL.NUM_TARGETS,
                                      entropy_lambda=cfg.MODEL.INFERENCE_ENTROPY_LAMBDA,
                                      non_target_lambda=cfg.MODEL.NON_TARGET_LAMBDA,
                                      auto_regression=cfg.TRAIN.AUTOREGRESSION).cuda()
    generative_condition_model_ = GenerativeModelConditionDistribution(num_targets=cfg.MODEL.NUM_TARGETS,
                                                                       tvs_dim=cfg.MODEL.TVS_DIM,
                                                                       use_stds=cfg.TRAIN.USE_STDS,
                                                                       num_processes=cfg.TRAIN.NUM_PROCESSES,
                                                                       mode=cfg.MODEL.INTERPOLATION_MODE,
                                                                       diff_lambda=cfg.MODEL.RECONSTRUCTION_DIFF_LAMBDA,
                                                                       diff_n=cfg.MODEL.RECONSTRUCTION_DIFF_N)

    trained_model_path = '/data1/huangqinlong/PhonemeExtraction/log_dir/' \
                         'train-Use_Stds-lr=0.001-entropy_lambda=0.1-DynamicRNN-targets=64-interpolation=linear-' \
                         'hidden=512+256+2-CrossEntropy-Tvs-num=15-diff=0.0-1-ntl=0.0-20210529-11:07:08'
    trained_model_pkl = 'WS_MODEL_Epoch100_recognition_loss=0.0305_condition_loss=0.1623_prior_loss=1.3014.pkl'
    trained_model_fp = os.path.join(trained_model_path, trained_model_pkl)

    checkpoint = torch.load(trained_model_fp)
    inference_model_.load_state_dict(checkpoint['inference_model_state_dict'])
    generative_condition_model_.load_state_dict(checkpoint['generative_condition_model_state_dict'])
    print('Loaded model from {}'.format(trained_model_fp))

    data_root = '/data1/huangqinlong/PhonemeExtraction/data'
    # # extract_tvs_for_generating(inference_model_, generative_condition_model_, ref_path=os.path.join(data_root, 'eval'),
    # #           source_dir=os.path.join(data_root, 'eval'),
    # #           target_dir=os.path.join(data_root, 'test', '20210424-00:43:21'))
    #
    ori_pred_fp = os.path.join(data_root, 'test', '64targets-20210529-11:07:08-zlogits-tvs-tvs_pred.pkl')
    inference(inference_model_, generative_condition_model_, os.path.join(data_root, 'train'),
              ori_pred_fp)

    fps = [
        # '64targets-20210529-11:05:26-tvs-tvs_pred.pkl',
        # '128targets-20210529-11:05:50-tvs-tvs_pred.pkl',
        # '500targets-20210513-22:03:13-tvs-tvs_pred.pkl',
        # '1000targets-20210512-22:09:35-tvs-tvs_pred.pkl',
        # '500targets-linear-20210511-17:02:51-tvs-tvs_pred.pkl',
        '64targets-20210529-11:07:08-zlogits-tvs-tvs_pred.pkl'
    ]

    color_list = ['red', 'blue']

    # fig = plt.figure(constrained_layout=True, figsize=(14, 20))
    # sub_figs = fig.subfigures(2, 2)
    # sub_fig_titles = ['(a)', '(b)', '(c)', '(d)']
    # for fp, sub_fig in zip(fps, sub_figs.flat):
    #     ori_pred_fp = os.path.join(data_root, 'test', fp)
    #     with open(ori_pred_fp, 'rb') as fr:
    #         tvs_list = pickle.load(fr)[15]
    #     tvs_num = tvs_list.shape[0]
    #     axs = sub_fig.subplots(len(tvs_dim_name), 1, sharex=True)
    #     # sub_fig.suptitle(t=title, x=0.55, y=0.0)
    #     for tvs_dim, ax in enumerate(axs):
    #         ax.plot(range(len(tvs_list[0])), tvs_list[0][:, tvs_dim], color=color_list[0], linestyle='solid')
    #         ax.plot(range(len(tvs_list[1])), tvs_list[1][:, tvs_dim], color=color_list[1], linestyle='dashed')
    #         ax.set_ylabel(tvs_dim_name[tvs_dim])
    #         ax.set_xlim(xmin=0)

    # for fp in fps:
    #     ori_pred_fp = os.path.join(data_root, 'test', fp)
    #     with open(ori_pred_fp, 'rb') as fr:
    #         tvs_list = pickle.load(fr)[21]
    #         fig = plot_single_ori_rec_tvs(tvs_list)  # 4, 6
    #     fig.show()

    for fp in fps:
        ori_pred_fp = os.path.join(data_root, 'test', fp)
        with open(ori_pred_fp, 'rb') as fr:
            z_logits, tvs_list = pickle.load(fr)[21]
            z_probs = np.exp(z_logits)/sum(np.exp(z_logits))
            # temp =
            # insert_vector = np.array(
            #     [[1.0 if temp[i].sum() > 0 else 0.0 for i in range(cfg.MODEL.NUM_TARGETS)]])  # [1, num_targets]
            # heatmap_array = np.concatenate([temp.T, insert_vector])  # [num_frames + 1, num_targets]
            fig = utils.plot_heat_map(z_probs, None)
            fig.show()
            fig = plot_single_ori_rec_tvs(tvs_list)  # 4, 6
            fig.show()


