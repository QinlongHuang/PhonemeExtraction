# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/data.py
# @Author: Qinlong Huang
# @Create Date: 2021/04/09 13:23
# @Contact: qinlonghuang@gmail.com
# @Description:

import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import glob
import numpy as np
import csv
from src.config import cfg
from utils.mocha_timit_preprocess import get_corpus_name, arti_not_available


def load_tvs_pkl(fp):

    with open(fp, 'rb') as fr:
        tvs = torch.from_numpy(pickle.load(fr)[-2])

    tvs_left = tvs[:, :7]
    tvs_right = tvs[:, 8:]
    tvs_without_r0 = torch.cat([tvs_left, tvs_right], dim=-1)

    return tvs_without_r0


def load_tvs_track_pkl(fp):
    
    with open(fp, 'rb') as fr:
        wave, sr, spec, tvs, track = pickle.load(fr)
        
    tvs_left = tvs[:, :7]
    tvs_right = tvs[:, 8:]
    tvs_without_r0 = np.concatenate([tvs_left, tvs_right], axis=-1)
    
    return track, tvs_without_r0

    
def read_csv_arti_ok_per_speaker(speaker_csv_fp):
    """
    create a dictionnary , with different categories as keys (from A to F).
    For a category the value is another dictionnary {"articulators" : list of 18 digit with 1 if arti is
    available for this category,"speakers" : list of speakers in this category}
    The dict is created based on the csv file "articulators_per_speaer"
    """
    arti_per_speaker = speaker_csv_fp
    csv.register_dialect('myDialect', delimiter=';')
    categ_of_speakers = dict()
    with open(arti_per_speaker, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for categ in ["A", "B", "C", "D", "E", "F"]:
            categ_of_speakers[categ] = dict()
            categ_of_speakers[categ]["sp"] = []
            categ_of_speakers[categ]["arti"] = None
        for row in reader:
            categ_of_speakers[row[19]]["sp"].append(row[0])
            if categ_of_speakers[row[19]]["arti"]  :
                if categ_of_speakers[row[19]]["arti"] != row[1:19]:
                    print("check arti and category for categ {}".format(row[19]))
            else:
                categ_of_speakers[row[19]]["arti"] = row[1:19]
    # for cle in categ_of_speakers.keys():
    #     print("categ ",cle)
    #     print(categ_of_speakers[cle])

    # Remove category
    speaker_arti_dict = dict()
    for cle in categ_of_speakers.keys():
        for sp in categ_of_speakers[cle]['sp']:
            speaker_arti_dict[sp] = categ_of_speakers[cle]['arti']
    return speaker_arti_dict

speaker_arti_csv = '/data1/huangqinlong/PhonemeExtraction/data/articulators_per_speaker.csv'
speaker_arti_dict = read_csv_arti_ok_per_speaker(speaker_arti_csv)


class M01_Dataset(Dataset):

    def __init__(self, ema_dir, sp='M01'):
        super(M01_Dataset, self).__init__()

        self.ema_fps = glob.glob(os.path.join(ema_dir, '*.npy'))
        arti = speaker_arti_dict[sp]
        self.used_index = list()
        for i, a in enumerate(arti):
            if a == '1':
                self.used_index.append(i)

    def __getitem__(self, index):
        ema = torch.from_numpy(np.load(self.ema_fps[index])).float()[:, self.used_index]
        num_frame = ema.shape[0]

        return ema, num_frame

    def __len__(self):
        return len(self.ema_fps)


class EMADataset(Dataset):

    def __init__(self, speaker_name):
        super(EMADataset, self).__init__()

        corpus = get_corpus_name(speaker_name)

        npy_path = cfg.DATA.data_root / corpus / 'processed' / speaker_name / 'ema_final'

        self.emas = list()
        fps = glob.glob(npy_path)

        delete_idxs = arti_not_available(speaker_name)

        # [tt_x|tt_y|td_x|td_y|tb_x|tb_y|li_x|li_y|ul_x|ul_y|ll_x|ll_y|la|pro|ttcl|tbcl|v_x|v_y]
        for fp in fps:
            ema = np.load(fps)
            ema = np.delete(ema, delete_idxs, axis=-1)
            self.emas.append(ema)

    def __getitem__(self, index):
        ema = torch.tensor(self.emas[index]).float()
        num_frame = ema.shape[0]
        return ema, num_frame

    def __len__(self):
        return len(self.emas)


class TvsDataset(Dataset):

    def __init__(self, fps_path, num_fps=5400):
        super(TvsDataset, self).__init__()

        self.tvs_fps = glob.glob(os.path.join(fps_path, '*.pkl'))[:num_fps]

    def __getitem__(self, index):

        # TODO: 到底用float还是double，如果用double，就去改模型参数的dtype
        tvs = load_tvs_pkl(self.tvs_fps[index]).float()
        num_frame = tvs.shape[0]

        return tvs, num_frame

    def __len__(self):

        return len(self.tvs_fps)


class PriorDataset(Dataset):

    def __init__(self, z_list):
        super(PriorDataset, self).__init__()

        self.zs = z_list

    def __getitem__(self, index):
        z = self.zs[index]
        num_frame = z.shape[0]

        return z, num_frame

    def __len__(self):
        return len(self.zs)
    

class TrackTvsDataset(Dataset):
    
    def __init__(self, tvs_fps):
        super(TrackTvsDataset, self).__init__()

        self.tvs_fps = tvs_fps

    def __getitem__(self, index):
        track, tvs = load_tvs_track_pkl(self.tvs_fps[index])
        track = torch.from_numpy(track)
        tvs = torch.from_numpy(tvs).float()
        num_frame = tvs.shape[0]

        return track, tvs, num_frame

    def __len__(self):
        return len(self.tvs_fps)


def collate_tvs(batch):
    """
    Build a batch from a given list
    :param batch: a list of tensors
    :return: batch: [batch_size, num_frames, obs_embedding_dim]
    """
    list_tvs, list_num_frame = map(list, zip(*batch))
    batch_tvs = pad_sequence(list_tvs, batch_first=True)
    batch_num_frame = torch.LongTensor(list_num_frame)

    return batch_tvs, batch_num_frame


def collate_track_tvs(batch):
    list_track, list_tvs, list_num_frame = map(list, zip(*batch))
    batch_track = torch.stack(list_track)
    batch_tvs = pad_sequence(list_tvs, batch_first=True)
    batch_num_frame = torch.LongTensor(list_num_frame)

    return batch_track, batch_tvs, batch_num_frame


if __name__ == '__main__':

    # path = '/data1/huangqinlong/PhonemeExtraction/data/eval/'  # 540 tvs
    path = '/data1/huangqinlong/PhonemeExtraction/data/train/'

    # tvs_list = list()
    #
    # num_frame_max = 1000
    # num_frame_factor_dist = torch.distributions.Uniform(0.3, 1)
    # for i in range(1000):
    #     tvs = torch.randn(int(num_frame_factor_dist.sample() * num_frame_max), 15)
    #     tvs_list.append(tvs)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # dataset = TvsDataset(path)
    # sample = dataset[123][0]
    # titles = ['microInt', 'glotVol', 'aspVol', 'fricVol', 'fricPos', 'fricCF', 'fricBW',
    #           'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'velum']
    # x = torch.linspace(0, len(sample) / 80, len(sample))
    # fig, axes = plt.subplots(figsize=(6, len(titles) * 1), ncols=1, nrows=len(titles))
    # for i, (ax, title) in enumerate(zip(axes, titles)):
    #     ax.plot(x, sample[:, i], color='green')
    #     ax.set_title(title)
    # plt.tight_layout()
    # fig.show()
    #
    # print(len(dataset))

    # fp = '/data1/huangqinlong/PhonemeExtraction/data/mocha-timit/msak0_v1.1_ema_12_lowpass=0.3_resample_rate=80.pkl'
    # dataset = Mocha_TIMITDataset(fp)

    # sample = dataset[23][0]
    # x = torch.linspace(0, len(sample) / 80, len(sample))
    # titles = ['lower incisor x', 'upper lip x', 'lower lip x', 'tongue tip x',
    #           'lower incisor y', 'upper lip y', 'lower lip y', 'tongue tip y',
    #           'tongue blade x', 'tongue dorsum x',
    #           'tongue blade y', 'tongue dorsum y']
    # fig, axes = plt.subplots(figsize=(6, len(titles) * 1), ncols=1, nrows=len(titles))
    # for i, (ax, title) in enumerate(zip(axes, titles)):
    #     ax.plot(x, sample[:, i], color='green')
    #     ax.set_title(title)
    # plt.tight_layout()
    # fig.show()
    #
    # sns.distplot([data[0].shape[0] for data in dataset])
    # plt.show()

    ema_dir = '/data1/huangqinlong/PhonemeExtraction/data/M01/ema_final'
    dataset = M01_Dataset(ema_dir)
    print(len(dataset))
    # data = dataset[0][0]
    # titles = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y', 'ul_x', 'ul_y', 'll_x', 'll_y', 'la',
    #          'pro', 'ttcl', 'tbcl']
    # fig, axes = plt.subplots(figsize=(6, 16), nrows=16, ncols=1)
    # for i, ax in enumerate(axes):
    #     ax.plot(data[:, i])
    #     ax.set_title(titles[i])
    # plt.tight_layout()
    # fig.show()

    # datas = np.random.normal(140, 18, 500)
    # datas_ = np.random.normal(210, 28, 500)
    # datas = np.concatenate([datas, datas_])
    # sns.distplot([data for data in datas])

    # sns.distplot([data[0].shape[0] for data in dataset])
    # plt.show()

