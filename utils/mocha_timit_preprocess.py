################################################################################
# -*- coding: utf-8 -*-                                                        #
# @File: mocha_timit_preprocess.py                                             #
# Project: utils                                                               #
# Create Time: 2021/09/23 12:02:09                                             #
# Author: Huang Qinlong                                                        #
# -----                                                                        #
# Last Modified: Thu Sep 23 2021                                               #
# Modified By: Huang Qinlong                                                   #
# -----                                                                        #
# HISTORY:                                                                     #
# Date      	By	Comments                                                   #
# ----------	---	---------------------------------------------------------  #
################################################################################
from operator import index
import os
import pickle
from multiprocessing import Pool
import csv

import numpy as np
import seaborn as sns
import glob
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from torch.functional import norm
from tqdm import tqdm

from src.config import cfg

def get_corpus_name(speaker_name):

    if speaker_name == "MNGU0":
        corpus = "MNGU0"
    elif speaker_name in ["F1", "F5", "M1", "M3"]:
        corpus = "usc"
    elif speaker_name in ["F01", "F02", "F03", "F04", "M01", "M02", "M03", "M04"]:
        corpus = "Haskins"
    elif speaker_name in  ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]:
        corpus = "mocha"
    else:
        raise NameError("Please check your speaker name!")
    return corpus

def arti_not_available(speaker_name):
    csv.register_dialect('myDialect', delimiter=';')
    with open(cfg.DATA.speaker_arti_csv, 'r') as csvFile:
        reader = csv.reader(csvFile, dialect="myDialect")
        next(reader)
        for row in reader:
            if row[0] == speaker_name: # we look for our speaker
                arti_to_consider = row[1:19]  # 1 if available
    arti_not_avail = [k for k, n in enumerate(arti_to_consider) if n == "0"] # 1 of NOT available

    return arti_not_avail

class Speaker(object):

    def __init__(self, speaker_name):

        super(Speaker, self).__init__()

        self._speaker_name = speaker_name
        assert speaker_name in ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]
        self.corpus = get_corpus_name(speaker_name)

        self._raw_path = cfg.DATA.mocha_root / 'raw' / speaker_name
        self._processed_path = cfg.DATA.mocha_root / 'processed' / speaker_name
        if not os.path.exists(self._processed_path):
            os.mkdir(self._processed_path)
        self.EMA_files = sorted([name for name in os.listdir(self._raw_path) if "palate" not in name])
        self.EMA_files = sorted([name[:-4] for name in self.EMA_files if name.endswith('.ema')])
        self.n_columns = 20

        self.articulators = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y', 'li_x', 'li_y', 'ul_x', 'ul_y', 'll_x', 'll_y']
        self.speakers_with_velum = ["fsew0", "msak0", "faet0", "ffes0", "falh0"]
        if speaker_name in self.speakers_with_velum:
            self.articulators.extend(['v_x', 'v_y'])

        self.list_EMA_traj = list()
        self.std_ema = None
        self.moving_average_ema = None
        self.mean_ema = None

        self.get_corpus_name()
        self.init_corpus_param()

        self.hop_time = 10 / 1000
        
    def create_missing_dir(self):
        """
        delete all previous preprocessing, create needed directories
        """
        if not os.path.exists(os.path.join(self._processed_path, "ema")):
            os.makedirs(os.path.join(self._processed_path, "ema"))
        if not os.path.exists(os.path.join(self._processed_path, "ema_final")):
            os.makedirs(os.path.join(self._processed_path, "ema_final"))

        files = glob.glob(os.path.join(self._processed_path, "ema", "*"))
        files += glob.glob(os.path.join(self._processed_path, "ema_final", "*"))

        for f in files:
            os.remove(f)

    def init_corpus_param(self):

        if self.corpus == "mocha":
            self.sampling_rate_wav = 16000
            self.sampling_rate_ema = 500
            self.cutoff = 10
        elif self.corpus == "MNGU0":
            self.sampling_rate_wav = 16000
            self.sampling_rate_ema = 200
            self.cutoff = 10

        elif self.corpus == "usc":
            self.sampling_rate_wav = 20000
            self.sampling_rate_ema = 100
            self.cutoff = 10

        elif self.corpus == "Haskins":
            self.sampling_rate_wav = 44100
            self.sampling_rate_ema = 100
            self.cutoff = 20
        else:
            raise NameError("Please check your speaker name!")

    @staticmethod
    def _low_pass_filter(cut_off, sampling_rate):
        """
        :param cut_off:  cutoff of the filter
        :param sampling_rate:  sampling rate of the data 
        :return: the weights of the lowpass filter
        implementation of the weights of a low pass filter windowed with a hanning winow.
        The filter is normalized (gain 1)
        """
        fc = cut_off/sampling_rate  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
        if fc > 0.5:
            raise Exception("The cut-off frequency must be at least twice the sampling frequency")
        b = 0.08   # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b))) # window
        if not N % 2:
            N += 1  # Make sure that N is odd.
        n = np.arange(N)
        h = np.sinc(2 * fc * (n - (N - 1) / 2))  # Compute sinc filter.
        w = 0.5 * (1 - np.cos(2 * np.pi * n / (N-1)))  # Compute hanning window.
        h = h * w  # Multiply sinc filter with window.
        h = h / np.sum(h)
        return h

    def smooth_data(self, ema, sr=0):

        pad = 30
        if sr == 0:
            sr = self.sampling_rate_ema
        cutoff = self.cutoff
        weights = self._low_pass_filter(cutoff, sr)

        my_ema_filtered = np.concatenate([np.expand_dims(np.pad(ema[:, k], (pad, pad), "symmetric"), 1)
                                            for k in range(ema.shape[1])], axis=1)
        my_ema_filtered = np.concatenate([np.expand_dims(np.convolve(channel, weights, mode='same'), 1)
                                            for channel in my_ema_filtered.T], axis=1)
        my_ema_filtered = my_ema_filtered[pad:-pad, :]

        return my_ema_filtered


    def add_vocal_tract(self, my_ema):
        """
        calculate 4 'vocal tract' and reorganize the data into a 18 trajectories in a precised order
        :param my_ema: EMA trajectory with K points
        :return: a np array (18,K) where the trajectories are sorted, and unavailable trajectories are at 0
        """

        def add_lip_aperture(ema):
            """
            唇孔半径
            lip aperture trajectory equals "upperlip_y - lowerlip_y"
            """
            ind_1, ind_2 = [self.articulators.index("ul_y"), self.articulators.index("ll_y")]
            lip_aperture = ema[:, ind_1] - ema[:, ind_2]
            return lip_aperture

        def add_lip_protrusion(ema):
            """
            唇凸
            lip protrusion trajectory (upperlip_x + lowerlip_x) / 2
            """
            ind_1, ind_2 = [self.articulators.index("ul_x"), self.articulators.index("ll_x")]
            lip_protrusion = (ema[:, ind_1] + ema[:, ind_2]) / 2
            return lip_protrusion

        def add_TTCL(ema):
            """
            tongue tip constriction location
            """
            ind_1, ind_2 = [self.articulators.index("tt_x"), self.articulators.index("tt_y")]
            TTCL = ema[:, ind_1] / np.sqrt(ema[:, ind_1] ** 2 + ema[:, ind_2] ** 2)
            return TTCL

        def add_TBCL(ema):
            """
            tongue body constriction location
            """
            ind_1, ind_2 = [self.articulators.index("tb_x"), self.articulators.index("tb_y")]
            TBCL = ema[:, ind_1] / np.sqrt(ema[:, ind_1] ** 2 + ema[:, ind_2] ** 2)
            return TBCL

        lip_aperture = add_lip_aperture(my_ema)
        lip_protrusion = add_lip_protrusion(my_ema)
        TTCL = add_TTCL(my_ema)
        TBCL = add_TBCL(my_ema)

        if self._speaker_name in self.speakers_with_velum:
            my_ema = np.concatenate((my_ema, np.zeros((len(my_ema), 4))), axis=1)
            my_ema[:, 16:18] = my_ema[:, 12:14]
            my_ema[:, 12:16] = 0
        else:
            my_ema = np.concatenate((my_ema, np.zeros((len(my_ema), 6))), axis=1)

        my_ema[:, 12] = lip_aperture
        my_ema[:, 13] = lip_protrusion
        my_ema[:, 14] = TTCL
        my_ema[:, 15] = TBCL
        idx_to_ignore = arti_not_available()
        my_ema[:, idx_to_ignore] = 0

        return my_ema

    def calculate_norm_values(self):
        """
        based on all the EMA trajectories calculate the norm values :
        - mean of ema
        - std of ema
        - moving average for ema on 60 sentences
        then save those norm values
        """
        list_EMA_traj = self.list_EMA_traj

        pad = 30
        all_mean_ema = np.array([np.mean(traj, axis=0) for traj in list_EMA_traj]) # (18, n_sentences)
        # np.save(os.path.join("norm_values", "all_mean_ema_" + self._speaker_name), all_mean_ema)
        #    weights_moving_average = low_pass_filter_weight(cut_off=10, sampling_rate=self.sampling_rate_ema)
        all_mean_ema = np.concatenate([np.expand_dims(np.pad(all_mean_ema[:, k], (pad, pad), "symmetric"), 1)
                                        for k in range(all_mean_ema.shape[1])], axis=1)

        moving_average = np.array(
            [np.mean(all_mean_ema[k - pad:k + pad], axis=0) for k in range(pad, len(all_mean_ema) - pad)])

        all_EMA_concat = np.concatenate([traj for traj in list_EMA_traj], axis=0)
        std_ema = np.std(all_EMA_concat, axis=0)
        std_ema[std_ema < 1e-3] = 1

        mean_ema = np.mean(np.array([np.mean(traj, axis=0) for traj in list_EMA_traj]), axis=0)

        # np.save(os.path.join("norm_values", "moving_average_ema_" + self._speaker_name), moving_average)
        # np.save(os.path.join("norm_values", "std_ema_" + self._speaker_name), std_ema)
        # np.save(os.path.join("norm_values", "mean_ema_" + self._speaker_name), mean_ema)

        self.std_ema = std_ema
        self.moving_average_ema = moving_average
        self.mean_ema = mean_ema

    def normalize_sentence(self, i, my_ema_filtered):
        """
        :param i: index of the ema traj (to get the moving average)
        :param my_ema_filtered: the ema smoothed ema traj
        :return: the normalized EMA data
        """
        my_ema_VT = (my_ema_filtered - self.moving_average_ema[i, :]) / self.std_ema

        return my_ema_VT

    def read_ema_file(self, k):
        """
        Read ema file from MOCHA-TIMIT dataset
        MOCHA means MultiCHannel Articulatory
        """
        path_ema_file = os.path.join(self._raw_path, self.EMA_files[k] + ".ema")
        with open(path_ema_file, 'rb') as ema_annotation:
            column_names = [0] * self.n_columns
            for line in ema_annotation:
                line = line.decode('latin-1').strip("\n")
                if line == 'EST_Header_End':
                    break
                elif line.startswith('NumFrames'):
                    n_frames = int(line.rsplit(' ', 1)[-1])
                elif line.startswith('Channel_'):
                    col_id, col_name = line.split(' ', 1)
                    column_names[int(col_id.split('_', 1)[-1])] = col_name.replace(" ","")  # v_x has sometimes a space
            ema_data = np.fromfile(ema_annotation, "float32").reshape(n_frames, -1)
            cols_index = [column_names.index(col) for col in self.articulators]
            ema_data = ema_data[:, cols_index]
            ema_data = ema_data / 100  # met en mm, initallement en 10^-1m
            if np.isnan(ema_data).sum() != 0:
                print("Num of nan: ", np.isnan(ema_data).sum())
                # Build a cubic spline out of non-NaN values.
                spline = interpolate.splrep(np.argwhere(~np.isnan(ema_data).ravel()),
                                                    ema_data[~np.isnan(ema_data)], k=3)
                # Interpolate missing values and replace them.
                for j in np.argwhere(np.isnan(ema_data)).ravel():
                    ema_data[j] = interpolate.splev(j, spline)

            return ema_data

    def preprocess_speaker(self, normalization='minmax'):

        assert normalization in ['minmax', 'standardization']

        self.create_missing_dir()

        N = len(self.EMA_files)

        if normalization == 'minmax':
            temp_max = list()
            temp_min = list()
            max_total = None
            min_total = None
        for i in tqdm(range(N)):
            ema = self.read_ema_file(i)
            ema_VT = self.add_vocal_tract(ema)
            ema_VT_smooth = self.smooth_data(ema_VT)  # smooth for a better calculation of norm values

            np.save(self._processed_path / "ema" / self.EMA_files[i], ema_VT)
            np.save(self._processed_path / "ema_final" / self.EMA_files[i], ema_VT_smooth)
            self.list_EMA_traj.append(ema_VT_smooth)

            if normalization == 'minmax':
                temp_max.append(ema_VT_smooth.max(axis=0))
                temp_min.append(ema_VT_smooth.min(axis=0))
                max_per_sample = np.array(temp_max)
                min_per_sample = np.array(temp_min)
                max_total = max_per_sample.max(axis=0)
                min_total = min_per_sample.min(axis=0)

        if normalization == 'standardization':
            self.calculate_norm_values()
        elif normalization == 'minmax':
            b = 1
            a = -1
        for i in tqdm(range(N)):
            ema_pas_smooth = np.load(self._processed_path / "ema" / (self.EMA_files[i] + ".npy"))
            ema_VT_smooth = np.load(self._processed_path / "ema_final" / (self.EMA_files[i] + ".npy"))
            if normalization == 'minmax':
                ema_pas_smooth_norma = (b-a) * (ema_pas_smooth - min_total) / (max_total - min_total) + a
                ema_VT_smooth_norma = (b-a) * (ema_VT_smooth - min_total) / (max_total - min_total) + a
            elif normalization == 'standardization':
                ema_VT_smooth_norma = self.normalize_sentence(i, ema_VT_smooth)
                ema_pas_smooth_norma = self.normalize_sentence(i, ema_pas_smooth)
            new_sr = 1 / self.hop_time   # we did undersampling of ema traj for 1 point per frame mfcc
                                        # so about 1 point every hoptime sec.
            ema_VT_smooth_norma = self.smooth_data(ema_VT_smooth_norma, new_sr)
            np.save(self._processed_path / "ema" / self.EMA_files[i], ema_pas_smooth_norma)
            np.save(self._processed_path / "ema_final" / self.EMA_files[i], ema_VT_smooth_norma)

        print("Speaker " + self._speaker_name + " has been processed!")

    # def read_a_ema_file(self, path): 
        # def clean(s):
        #     return s.rstrip('\n').strip()

        # # parsing EMA files
        # columns = dict()
        # columns[0] = 'time'
        # columns[1] = 'present'

        # with open(path, 'rb') as fr:
        #     fr.readline()  # EST_File Track
        #     # decode: bytes->str, encode: str->bytes
        #     datatype = clean(fr.readline().decode()).split()[1]
        #     nframes = int(clean(fr.readline().decode()).split()[1])
        #     fr.readline()  # ByteOrder
        #     nchannels = int(clean(fr.readline().decode()).split()[1])
        #     while 'CommentChar' not in fr.readline().decode():
        #         pass  # EqualSpace, BreaksPresent, CommentChar
        #     fr.readline()  # empty line

        #     line = clean(fr.readline().decode())
        #     while "EST_Header_End" not in line:
        #         channel_number = int(line.split()[0].split('_')[1])
        #         channel_name = line.split()[1]
        #         columns[channel_number + 2] = channel_name
        #         line = clean(fr.readline().decode())
        #     # print("Columns: {}\ndatatype: {}\nnframes: {}\nnchannels: {}".format(columns, datatype, nframes, nchannels))

        #     string = fr.read()
        #     data = np.frombuffer(string, dtype='float32')
        #     data = np.reshape(data, (-1, len(columns)))  # [nframes, columns]

        # assert (nframes == data.shape[0])
        # assert (data[:, 1].all())
        # assert ((np.abs(np.diff(data[:, 0]) - 2.0E-3) < 1.0E-6).all())  # 500 Hz

        # print(columns.items())
        # idxs = [k for k, v in columns.items() if ('ui' in v or 'bn' in v or 'v' in v or '*' in v)]
        # idxs.extend([0, 1])
        # data = np.delete(data, idxs, axis=-1)
        # assert (data.shape[1] == 12)

        # 降采样
        # titles = ['lower incisor x', 'upper lip x', 'lower lip x', 'tongue tip x',
        #           'lower incisor y', 'upper lip y', 'lower lip y', 'tongue tip y',
        #           'tongue blade x', 'tongue dorsum x',
        #           'tongue blade y', 'tongue dorsum y']
        # fig, axes = plt.subplots(figsize=(6, len(titles) * 1), ncols=1, nrows=len(titles))
        # time = np.linspace(0, len(data) / 500, len(data))
        # for i, (ax, title) in enumerate(zip(axes, titles)):
        #     ax.plot(time, data[:, i], color='green')
        #     ax.set_title(title)
        # for i, ax in enumerate(axes):
        #     y = signal.resample(data[:, i], round(len(data[:, i]) / 500 * resample_rate))
        #     b, a = signal.butter(8, 0.3, 'lowpass')
        #     filted_data = signal.filtfilt(b, a, y.T).T
        #     time = np.linspace(0, len(y) / resample_rate, len(y))
        #     ax.plot(time, filted_data, color='r')
        # plt.tight_layout()  # 调整布局，避免坐标轴和标题的overlap
        # fig.show()

        # resampled_data = signal.resample(data, round(len(data) / 500 * resample_rate))
        # b, a = signal.butter(8, 0.3, 'lowpass')
        # filted_data = signal.filtfilt(b, a, resampled_data.T).T

        # return filted_data

    # def read_ema_path(self, dir: str, resample_rate):
    #     fps = glob.glob(dir + '/*.ema')

    #     pool = Pool(processes=64)
    #     temp = list()
    #     for fp in fps:
    #         temp.append(pool.apply_async(self.read_a_ema_file, (fp, resample_rate)))
    #     emas = [t.get() for t in temp]

    #     return emas

    # def save_ema_to_pickle(self, dir: str, resample_rate, a=-1, b=1):
    #     speaker_paths = glob.glob(dir + '/*v1.1')
    #     for path in speaker_paths:
    #         emas = self.read_ema_path(path, resample_rate)

    #         # TODO：不同说话人的Normalization应该用同一个norm参数吗
    #         temp_max = list()
    #         temp_min = list()

    #         for ema in emas:
    #             temp_max.append(ema.max(axis=0))
            #     temp_min.append(ema.min(axis=0))

            # fig, axes = plt.subplots(figsize=(8, 20), nrows=12, ncols=2)
            # for i in range(12):
            #     sns.distplot([max_[i] for max_ in temp_max], ax=axes[i][0])
            #     sns.distplot([min_[i] for min_ in temp_min], ax=axes[i][1])
            # plt.tight_layout()
            # fig.show()

            # max_per_sample = np.array(temp_max)
            # min_per_sample = np.array(temp_min)
            # max_total = max_per_sample.max(axis=0)
            # min_total = min_per_sample.min(axis=0)

            # norm_emas = [(b-a) * (ema - min_total) / (max_total - min_total) + a for ema in emas]

            # filename = path.split('/')[-1] + '_ema_12_lowpass=0.3_resample_rate={}.pkl'.format(resample_rate)
            # with open(os.path.join(dir, filename), 'wb') as fw:
            #     pickle.dump(norm_emas, fw)
            # print('Save {} to {}'.format(filename, dir))


# def read_a_lab_file(path: str):
#     with open(path, 'rb') as fr:
#         lines = fr.readlines()
#         phoneme_set = set()
#         for line in lines:
#             phoneme = line.decode().split('\n')[0].split(' ')[-1]
#             phoneme_set.add(phoneme)
#             print(line)

#     return phoneme_set


# def read_lab_path(dir: str):
#     fps = glob.glob(dir + '/*.lab')

#     pool = Pool(processes=64)
#     temp = list()
#     for fp in fps:
#         temp.append(pool.apply_async(read_a_lab_file, (fp,)))
#     phoneme_sets = [t.get() for t in temp]

#     phoneme_set = set()
#     for phoneme_set_ in phoneme_sets:
#         phoneme_set.update(phoneme_set_)

#     print(len(phoneme_set))

# Usage
# lab_fp = '/data1/huangqinlong/PhonemeExtraction/data/mocha-timit/msak0_v1.1/msak0_001.lab'
# read_a_lab_file(lab_fp)

# timit_dir = '/data1/huangqinlong/PhonemeExtraction/data/mocha-timit/fsew0_v1.1'
# read_lab_path(timit_dir)

# speaker_names = ["fsew0", "msak0", "faet0", "ffes0", "maps0", "mjjn0", "falh0"]
# speaker = Speaker(speaker_names[1])

# speaker.preprocess_speaker(normalization='minmax')

# ema_smoothed = np.load(cfg.DATA.mocha_root / 'processed' / 'msak0' / 'ema_final' / 'msak0_378.npy')

# idxs = [5, 6, 8, 9, 12, 13, 15]
# ema_smoothed = np.delete(ema_smoothed, idxs, axis=-1)
# print(ema_smoothed.max())

