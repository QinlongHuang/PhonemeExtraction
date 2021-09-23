# -*- coding: utf-8 -*-
# @File: PhonemeExtraction/utils.py
# @Author: Qinlong Huang
# @Create Date: 2021/03/31 14:57
# @Contact: qinlonghuang@gmail.com
# @Description:


import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import torch
import math
from torch import nn
import random
import numpy as np
from multiprocessing.pool import Pool
import pickle
import glob
from scipy import signal
from src.config import cfg
import parselmouth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def logits_uniform_mixture(values, delta, dim=-1):
    """
    Given logits for a categorical distribution D, return the logits
    for a mixture distribution (1-delta)*D + delta*uniform
    """
    n = values.shape[dim]
    term1 = values + math.log(1 - delta)
    term2 = torch.logsumexp(values + math.log(delta / n), dim=dim, keepdim=True).expand_as(term1)
    logits = torch.stack([term1, term2]).logsumexp(dim=0)
    return logits


def save_pretrain_checkpoint(inference_model, optimizer_inference, log_dir, loss):
    checkpoint = {}

    data_p = isinstance(inference_model, nn.DataParallel)
    checkpoint['model_state_dict'] = inference_model.module.state_dict() if data_p else inference_model.state_dict()

    checkpoint_path = os.path.join(log_dir, "PRETRAINED_INFERENCE_MODEL_loss={:.4f}.pkl".format(loss))
    torch.save(checkpoint, checkpoint_path)
    print('Saved checkpoint into {} !'.format(checkpoint_path))


def save_training_checkpoint(inference_model, generative_prior_model, generative_condition_model,
                                optimizer_inference,
                                optimizer_generative_prior,
                                optimizer_generative_condition,
                                log_dir, epoch, recognition_loss, condition_loss, prior_loss):
    checkpoint = {}

    data_p = isinstance(inference_model, nn.DataParallel)
    checkpoint['inference_model_state_dict'] = inference_model.module.state_dict() if data_p else \
        inference_model.state_dict()
    checkpoint['generative_prior_model_state_dict'] = generative_prior_model.module.state_dict() if data_p else \
        generative_prior_model.state_dict()
    checkpoint['generative_condition_model_state_dict'] = generative_condition_model.state_dict()  # cpu only
    checkpoint['optimizer_inference_state_dict'] = optimizer_inference.state_dict()
    checkpoint['optimizer_generative_prior_state_dict'] = optimizer_generative_prior.state_dict()
    checkpoint['optimizer_generative_condition_state_dict'] = optimizer_generative_condition.state_dict()

    checkpoint_path = os.path.join(log_dir, "WS_MODEL_Epoch{}_"
                                            "recognition_loss={:.4f}_condition_loss={:.4f}_"
                                            "prior_loss={:.4f}.pkl".format(epoch, recognition_loss,
                                                                        condition_loss, prior_loss))
    torch.save(checkpoint, checkpoint_path)
    print('Saved checkpoint into {} !'.format(checkpoint_path))


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # if benchmark=True, deterministic will be False
        # if your model architecture and input size remains unchanged, then set benchmark = True.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=16, linewidth=400)  # profile='full'


def plot_tvs(tvs_list: list, z):
    """

    :param tvs_list: [tvs_num, seq_len, tvs_dim] np.ndarray
    :param z: [seq_len]
    :return:
    """
    color_list = ['green', 'red']
    tvs_num = len(tvs_list) if isinstance(tvs_list, list) else tvs_list.shape[0]
    assert (tvs_num == 1 or tvs_num == 2)
    sub_num = 1 if tvs_num == 1 else 3
    fig = plt.figure(figsize=(5 * sub_num, 8))
    tvs_list = [np.concatenate([z[:, None], tvs], axis=-1) for tvs in tvs_list]  # [tvs_num, seq_len, 1 + tvs_dim]

    for tvs_idx in range(tvs_num):
        for tvs_dim in range(cfg.MODEL.TVS_DIM + 1):
            ax = fig.add_subplot(cfg.MODEL.TVS_DIM + 1, sub_num, sub_num * tvs_dim + tvs_idx + 1)
            ax.plot(range(len(tvs_list[tvs_idx])), tvs_list[tvs_idx][:, tvs_dim], color=color_list[tvs_idx])
            # ax.set_yticks([])
            # ax.set_yticks([np.max(tvs_list[tvs_idx, :, tvs_dim]), np.min(tvs_list[tvs_idx, :, tvs_dim])])
    if sub_num == 3:
        for tvs_dim in range(cfg.MODEL.TVS_DIM + 1):
            ax = fig.add_subplot(cfg.MODEL.TVS_DIM + 1, sub_num, sub_num * tvs_dim + sub_num)
            ax.plot(range(len(tvs_list[0])), tvs_list[0][:, tvs_dim], color=color_list[0])
            ax.plot(range(len(tvs_list[1])), tvs_list[1][:, tvs_dim], color=color_list[1])
            # ax.set_yticks([])
            # ax.set_yticks([np.max(tvs_list[tvs_idx, :, tvs_dim]), np.min(tvs_list[tvs_idx, :, tvs_dim])])

    return fig


def plot_heat_map(z_logit, z_pred):
    """
    Plot heatmaps of z_logit and z_pred
    :param z_logit: [num_frame, num_targets]
    :param z_pred: [num_frame, num_targets]
    :return: fig
    """

    fig, (ax1, ax2) = plt.subplots(figsize=(12 * 1.5, 10 * 1.5), nrows=2, sharex=True)
    ax1.set_title('Heatmap of z_logits')
    ax2.set_title('Heatmap of z_preds')
    sns.heatmap(z_logit.T, ax=ax1, cmap="RdBu_r")
    # sns.heatmap(z_pred.T, ax=ax2, cmap="Greys")

    return fig


def plot_error_bar(means, stds):
    # Subplots
    # fig, axes = plt.subplots(figsize=(15 * 2, 10 * 2), nrows=8, ncols=8, sharex=True, sharey=True)
    # x = ['microInt', 'glotVol', 'aspVol', 'fricVol', 'fricPos', 'fricCF', 'fricBW',
    #      'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'velum']
    # for i, axs in enumerate(axes):
    #     for j, ax in enumerate(axs):
    #         sns.barplot(x=x, y=means[i * 8 + j].tolist(), yerr=stds[i * 8 + j].tolist(), ax=ax, palette="Blues_d",
    #                     errwidth=0.5)

    # fig = plt.figure(figsize=(10, 6))
    # sns.barplot(x=x, y=means.tolist(), yerr=stds.tolist(), palette="Blues_d", errwidth=2)

    fig, ax = plt.subplots(figsize=(12 * 5, 10))

    # tvs dataset
    y = ['microInt', 'glotVol', 'aspVol', 'fricVol', 'fricPos', 'fricCF', 'fricBW',
         'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'velum']

    # mocha-timit
    # y = ['lower incisor x', 'upper lip x', 'lower lip x', 'tongue tip x',
    #      'lower incisor y', 'upper lip y', 'lower lip y', 'tongue tip y',
    #      'tongue blade x', 'tongue dorsum x',
    #      'tongue blade y', 'tongue dorsum y']

    sns.set(style="whitegrid")
    x_major_locator = plt.MultipleLocator(3)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xlim(-3, cfg.MODEL.NUM_TARGETS * 3)

    for i in range(cfg.MODEL.NUM_TARGETS):
        ax.errorbar(x=means[i] + i * 3, y=y, xerr=stds[i], ecolor='r', color='b', elinewidth=2,
                    capsize=4)
        ax.axvline(x=0 + 3 * i, color='r', linestyle='--')
    return fig


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize_tvs(tvs):
    _spec_min = -16
    _spec_max = 8  # King-ASR 6, Sitaduoji 9, lvsongshi 7, default 10, ljspeech 8

    _tvs_min = np.array([-30.0, 0.0, 0.0, 0.0, 0.0, 500.0, 500.0, 0.8, 0.2, 0.2, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0],
                        dtype=np.float32)
    _tvs_max = np.array([17.0, 60.0, 20.0, 1.5, 7.0, 7000.0, 5000.0, 0.8, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 1.0],
                        dtype=np.float32)

    _track_min = np.array([55, 8, -24, 30, 10, 15, 0.5, 0.2, 2, 2000, 2000, 0.5, 1200], dtype=np.float32)
    _track_max = np.array([60, 20, 20, 45, 35, 40, 8, 5, 10, 6000, 6000, 10, 1800], dtype=np.float32)
    _TINY = 1e-8

    tvs = np.clip(tvs, _tvs_min, _tvs_max)
    tvs_norm = (tvs - _tvs_min) / (_tvs_max - _tvs_min + _TINY)
    tvs_norm = tvs_norm * 2 - 1  # rerange to [-1, 1]
    return tvs_norm


class Pchip(object):
    """
    不保单调Pchip的numpy版本
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @staticmethod
    def _find_derivatives(x: np.ndarray, y: np.ndarray):
        """
        用分段三次Hermite插值多项式算法计算(x_k, y_k)点处的导数值d_k
        记m_k为第k段([x_k, x_{k+1}], k=0, 1, 2, ..., n-1)的斜率(slope)
        另记h_k = x_{k+1} - x_k，即第k段的长度
        如果m_k=0或m_{k-1}=0，或m_k与m_{k-1}异号，则令d_k=0
        否则使用加权调和均值(Weighted Harmonic Mean)
        w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
        1/d_k = 1/(w_1 + w_2) * (w_1 / m_k + w_2 / m_{k-1})
        :param x: [n_steps, ]
        :param y: [n_steps, ...], e.g., [n_steps, 15]
        :return:
        """

        def _edge_case(h0, h1, m0, m1):
            # 处理两个端点
            # one-sided three-point estimate for the derivative
            d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)

            # try to preserve shape
            mask = np.sign(d) != np.sign(m0)
            mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d) > 3. * np.abs(m0))
            mmm = (~mask) & mask2

            d[mask] = 0.
            d[mmm] = 3. * m0[mmm]

            return d

        y_shape = y.shape
        if len(y.shape) == 1:
            # So that _edge_case doesn't end up assigning to scalars
            x = x[:, None]
            y = y[:, None]
        elif len(x.shape) < len(y.shape):
            x = x[:, None]

        hk = x[1:] - x[:-1]  # [n_steps - 1, ...]，记每一段的长度为h_k
        mk = (y[1:] - y[:-1]) / hk  # [n_step - 1, ...]，不会有x相同的点，计算每一段的斜率m_k

        if y.shape[0] == 2:  # 如果一共只有两个点，则进行线性插值
            dk = np.zeros_like(y)
            dk[0] = mk
            dk[1] = mk
            return dk.reshape(y_shape)

        # 考虑哪些点的导数要设为0
        # 1.计算每一段的斜率的符号
        # 2.第k段和第k+1段是否会满足d_k=0的条件
        smk = np.sign(mk)
        condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)  # m_k与m_{k-1}异号 or m_k=0 or m_{k-1}=0

        # w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
        # 1/d_k = 1/(w_1 + w_2) * (w_1 / m_k + w_2 / m_{k-1})
        w1 = 2*hk[1:] + hk[:-1]
        w2 = hk[1:] + 2*hk[:-1]

        # 这里mk可能出现0，会导致除0异常
        # 正向的时候会被condition处理掉，但是反向的时候不行
        whmean = (w1[~condition]/mk[~condition][:-1]
                  + w2[~condition]/mk[~condition][:-1]) / (w1[~condition] + w2[~condition])
        # whmean = (w1/mk[:-1] + w2/mk[1:]) / (w1+w2)

        # 不用调和平均值就可以避免这个问题
        # whmean = (w1*mk[:-1] + w2*mk[1:]) / (w1+w2)

        # 处理非端点
        dk = np.zeros_like(y)
        dk[1:-1][condition] = 0.0
        dk[1:-1][~condition] = 1.0 / whmean[~condition]

        # 端点应该特殊处理
        # as suggested in leve Moler, Numerical Computing with MATLAB, Chap 3.4
        dk[0] = _edge_case(hk[0], hk[1], mk[0], mk[1])
        dk[-1] = _edge_case(hk[-1], hk[-2], mk[-1], mk[-2])

        return dk.reshape(y_shape)

    @staticmethod
    def _h_poly(t: np.ndarray):
        """

        :param t: [new_nps]
        :return:  [4, new_nps]
        """
        tt = t[None] ** np.arange(4)[:, None]  # [4, new_nps]
        A = np.array([
            [1, 0, -3, 2],  # alpha_k(x)
            [0, 1, -2, 1],  # beta_k(x)
            [0, 0, 3, -2],  # alpha_{k+1}(x)
            [0, 0, -1, 1]  # beta_{k+1}(x)
        ], dtype=t.dtype)  # [4, 4]

        return A @ tt

    def interp(self, xs: np.ndarray):
        """
        x: [len(targets), ], y: [len(targets), 15]
        :param xs: [new_nps]
        :return:
        """
        # 计算每一点的一阶导
        # 首尾两个端点的导数和首尾两段的斜率保持一致，中间点的斜率为其前后两段斜率的平均
        # x_ = self.x[:, None]  # [len(targets), 1]
        # m = (self.y[1:] - self.y[:-1])/(x_[1:] - x_[:-1])  # 不会有x相同的点，计算每一段的斜率m_k, [n_points - 1]
        # m = np.concatenate([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])  # 计算每个点的导数, [n_points]
        d = Pchip._find_derivatives(self.x, self.y)

        idxs = np.searchsorted(self.x[1:], xs)  # idxs.shape == xs.shape，返回的是xs中的元素分别落在x中的哪一段, [new_nps]
        dx = (self.x[idxs + 1] - self.x[idxs])  # 计算xs中每个点所在的段的长度, [new_nps]
        # x[idxs]是xs所在段的左端点，xs-x[idxs]是xs与所在段的左端点的距离
        hh = (Pchip._h_poly((xs - self.x[idxs]) / dx))[..., None]  # [4, new_nps, 1]
        dx = dx[..., None]

        #   alpha_0(x) * y_0 + beta_0(x) * m_0      + alpha_1(x) * y_1 + beta_1(x) * m_1
        return hh[0] * self.y[idxs] + hh[1] * d[idxs] * dx + hh[2] * self.y[idxs + 1] + hh[3] * d[idxs + 1] * dx

    def __call__(self, xs: np.ndarray):
        return self.interp(xs)


class Tinterp1d(object):
    """
        A Pytorch implementation of scipy.interpolate.interp1d(kind='linear')
        """
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def interp(self, x_new):
        idxs = torch.searchsorted(self.x, x_new).clamp(1, len(self.x)-1)  # idxs.shape == xs.shape，返回的是xs中的元素分别落在x中的哪一段
        lo = idxs - 1
        hi = idxs
        m = (self.y[hi] - self.y[lo]) / (self.x[hi] - self.x[lo])[:, None]

        y_new = m * (x_new - self.x[lo])[:, None] + self.y[lo]

        return y_new

    def __call__(self, x_new):
        return self.interp(x_new)


class TPchip(object):
    """
    A Pytorch implementation of scipy.interpolate.PchipInterpolator
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, eps=1e-5):
        """

        :param x: [n_steps], A 1-D array of monotonically increasing real values. ``x`` cannot
        include duplicate values (otherwise f is overspecified)
        :param y: [n_steps, ...], A 1-D array of real values. ``y``'s length along the interpolation
        axis must be equal to the length of ``x``. If N-D array, use ``axis``
        parameter to select correct axis.
        """
        self.x = x
        self.y = y
        self.eps = eps

    @staticmethod
    def _find_derivatives(x: torch.Tensor, y: torch.Tensor, eps):
        """
        用分段三次Hermite插值多项式算法计算(x_k, y_k)点处的导数值d_k
        记m_k为第k段([x_k, x_{k+1}], k=0, 1, 2, ..., n-1)的斜率(slope)
        另记h_k = x_{k+1} - x_k，即第k段的长度
        如果m_k=0或m_{k-1}=0，或m_k与m_{k-1}异号，则令d_k=0
        否则使用加权调和均值(Weighted Harmonic Mean)
        w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
        1/d_k = 1/(w_1 + w_2) * (w_1 / m_k + w_2 / m_{k-1})
        :param x: [n_steps, ]
        :param y: [n_steps, ...], e.g., [n_steps, 15]
        :return:
        """

        def _edge_case(h0, h1, m0, m1):
            # 处理两个端点
            # one-sided three-point estimate for the derivative
            d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)

            # try to preserve shape
            mask = torch.sign(d) != torch.sign(m0)
            mask2 = (torch.sign(m0) != torch.sign(m1)) & (torch.abs(d) > 3. * torch.abs(m0))
            mmm = (~mask) & mask2

            d[mask] = 0.
            d[mmm] = 3. * m0[mmm]

            return d

        y_shape = y.shape
        if len(y.shape) == 1:
            # So that _edge_case doesn't end up assigning to scalars
            x = x[:, None]
            y = y[:, None]
        elif len(x.shape) < len(y.shape):
            x = x[:, None]

        hk = x[1:] - x[:-1]  # [n_steps - 1, ...]，记每一段的长度为h_k
        mk = (y[1:] - y[:-1]) / hk  # [n_step - 1, ...]，不会有x相同的点，计算每一段的斜率m_k

        if y.shape[0] == 2:  # 如果一共只有两个点，则进行线性插值
            dk = torch.zeros_like(y)
            dk[0] = mk
            dk[1] = mk
            return dk.reshape(y_shape)

        # 考虑哪些点的导数要设为0
        # 1.计算每一段的斜率的符号
        # 2.第k段和第k+1段是否会满足d_k=0的条件
        smk = torch.sign(mk)
        condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)  # m_k与m_{k-1}异号 or m_k=0 or m_{k-1}=0

        # w_1 = 2h_k + h_{k-1}, w_2 = h_k + 2h_{k-1}
        # 1/d_k = 1/(w_1 + w_2) * (w_1 / m_k + w_2 / m_{k-1})
        w1 = 2*hk[1:] + hk[:-1]
        w2 = hk[1:] + 2*hk[:-1]

        # 这里mk可能出现0，会导致除0异常
        # 正向的时候会被condition处理掉，但是反向的时候不行
        w1 = w1.expand(condition.shape)[~condition]
        w2 = w2.expand(condition.shape)[~condition]
        whmean = (w1/mk[:-1][~condition] + w2/mk[1:][~condition]) / (w1 + w2)
        # whmean = (w1/mk[:-1] + w2/mk[1:]) / (w1+w2)

        # 处理非端点
        dk = torch.zeros_like(y)
        dk[1:-1][condition] = 0.0
        dk[1:-1][~condition] = 1.0 / whmean

        # 端点应该特殊处理
        # as suggested in leve Moler, Numerical Computing with MATLAB, Chap 3.4
        dk[0] = _edge_case(hk[0], hk[1], mk[0], mk[1])
        dk[-1] = _edge_case(hk[-1], hk[-2], mk[-1], mk[-2])

        return dk.reshape(y_shape)

    @staticmethod
    def _h_poly(t: torch.Tensor):
        """

        :param t: [new_nps]
        :return:  [4, new_nps]
        """
        tt = t[None] ** torch.arange(4, device=t.device)[:, None]  # [4, new_nps]
        A = torch.tensor([
            [1, 0, -3, 2],  # alpha_k(x)
            [0, 1, -2, 1],  # beta_k(x)
            [0, 0, 3, -2],  # alpha_{k+1}(x)
            [0, 0, -1, 1]  # beta_{k+1}(x)
        ], dtype=t.dtype, device=t.device)  # [4, 4]

        return A @ tt

    def interp(self, x_new: torch.Tensor):
        """
        x: [len(targets), ], y: [len(targets), 15]
        :param x_new: [new_nps]
        :return:
        """
        # 计算每一点的一阶导
        # 1. 不保单调的PCHIP
        # 首尾两个端点的导数和首尾两段的斜率保持一致，中间点的斜率为其前后两段斜率的平均
        # x_ = self.x[:, None]  # [len(targets), 1]
        # m = (self.y[1:] - self.y[:-1])/(x_[1:] - x_[:-1])  # 不会有x相同的点，计算每一段的斜率m_k, [n_points - 1]
        # m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])  # 计算每个点的导数, [n_points]
        # 2. 保单调的PCHIP，在边界条件点的导数设为0，一般点处的导数设为其前后两段的加权调和平均
        d = TPchip._find_derivatives(self.x, self.y, self.eps)  # y.shape, [new_nps, ...]

        idxs = torch.searchsorted(self.x[1:], x_new)  # idxs.shape == xs.shape，返回的是xs中的元素分别落在x中的哪一段, [new_nps]
        dx = (self.x[idxs + 1] - self.x[idxs])  # 计算xs中每个点所在的段的长度, [new_nps]
        # x[idxs]是xs所在段的左端点，xs-x[idxs]是xs与所在段的左端点的距离
        hh = TPchip._h_poly((x_new - self.x[idxs]) / dx)[..., None]  # [4, new_nps, 1]
        dx.unsqueeze_(-1)

        #   alpha_0(x) * y_0 + beta_0(x) * m_0      + alpha_1(x) * y_1 + beta_1(x) * m_1
        y = hh[0] * self.y[idxs] + hh[1] * d[idxs] * dx + hh[2] * self.y[idxs + 1] + hh[3] * d[idxs + 1] * dx

        return y

    def __call__(self, xs: torch.Tensor):
        return self.interp(xs)


def read_a_ema_file(path: str, resample_rate=200):
    """
    Read ema file from MOCHA-TIMIT dataset
    MOCHA means MultiCHannel Articulatory
    460 utterances(~20 min)
    :param path:
    :return:
    """
    def clean(s):
        return s.rstrip('\n').strip()

    # parsing EMA files
    columns = {}
    columns[0] = 'time'
    columns[1] = 'present'

    with open(path, 'rb') as fr:
        fr.readline()  # EST_File Track
        # decode: bytes->str, encode: str->bytes
        datatype = clean(fr.readline().decode()).split()[1]
        nframes = int(clean(fr.readline().decode()).split()[1])
        fr.readline()  # ByteOrder
        nchannels = int(clean(fr.readline().decode()).split()[1])
        while 'CommentChar' not in fr.readline().decode():
            pass  # EqualSpace, BreaksPresent, CommentChar
        fr.readline()  # empty line

        line = clean(fr.readline().decode())
        while "EST_Header_End" not in line:
            channel_number = int(line.split()[0].split('_')[1])
            channel_name = line.split()[1]
            columns[channel_number + 2] = channel_name
            line = clean(fr.readline().decode())
        # print("Columns: {}\ndatatype: {}\nnframes: {}\nnchannels: {}".format(columns, datatype, nframes, nchannels))

        string = fr.read()
        data = np.frombuffer(string, dtype='float32')
        data = np.reshape(data, (-1, len(columns)))  # [nframes, columns]
    assert (nframes == data.shape[0])
    assert (data[:, 1].all())
    # np.diff(a, n=1, axis=-1)计算前后两个数据点的差，n代表阶数，即做n阶差分
    assert ((np.abs(np.diff(data[:, 0]) - 2.0E-3) < 1.0E-6).all())  # 500 Hz
    idxs = [k for k, v in columns.items() if ('ui' in v or 'bn' in v or 'v' in v or '*' in v)]
    idxs.extend([0, 1])
    data = np.delete(data, idxs, axis=-1)
    assert (data.shape[1] == 12)

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

    resampled_data = signal.resample(data, round(len(data) / 500 * resample_rate))
    b, a = signal.butter(8, 0.3, 'lowpass')
    filted_data = signal.filtfilt(b, a, resampled_data.T).T
    return filted_data


def read_ema_path(dir: str, resample_rate):
    fps = glob.glob(dir + '/*.ema')

    pool = Pool(processes=64)
    temp = list()
    for fp in fps:
        temp.append(pool.apply_async(read_a_ema_file, (fp, resample_rate)))
    pool.close()
    pool.join()
    emas = [t.get() for t in temp]

    return emas


def save_ema_to_pickle(dir: str, resample_rate, a=-1, b=1):
    speaker_paths = glob.glob(dir + '/*v1.1')
    for path in speaker_paths:
        emas = read_ema_path(path, resample_rate)

        # TODO：不同说话人的Normalization应该用同一个norm参数吗
        temp_max = list()
        temp_min = list()

        for ema in emas:
            temp_max.append(ema.max(axis=0))
            temp_min.append(ema.min(axis=0))

        fig, axes = plt.subplots(figsize=(8, 20), nrows=12, ncols=2)
        for i in range(12):
            sns.distplot([max_[i] for max_ in temp_max], ax=axes[i][0])
            sns.distplot([min_[i] for min_ in temp_min], ax=axes[i][1])
        plt.tight_layout()
        fig.show()

        max_per_sample = np.array(temp_max)
        min_per_sample = np.array(temp_min)
        max_total = max_per_sample.max(axis=0)
        min_total = min_per_sample.min(axis=0)

        norm_emas = [(b-a) * (ema - min_total) / (max_total - min_total) + a for ema in emas]

        filename = path.split('/')[-1] + '_ema_12_lowpass=0.3_resample_rate={}.pkl'.format(resample_rate)
        with open(os.path.join(dir, filename), 'wb') as fw:
            pickle.dump(norm_emas, fw)
        print('Save {} to {}'.format(filename, dir))


def read_a_lab_file(path: str):
    with open(path, 'rb') as fr:
        lines = fr.readlines()
        phoneme_set = set()
        for line in lines:
            phoneme = line.decode().split('\n')[0].split(' ')[-1]
            phoneme_set.add(phoneme)
            print(line)

    return phoneme_set


def read_lab_path(dir: str):
    fps = glob.glob(dir + '/*.lab')

    pool = Pool(processes=64)
    temp = list()
    for fp in fps:
        temp.append(pool.apply_async(read_a_lab_file, (fp,)))
    phoneme_sets = [t.get() for t in temp]

    phoneme_set = set()
    for phoneme_set_ in phoneme_sets:
        phoneme_set.update(phoneme_set_)

    print(len(phoneme_set))


def diff(t: torch.Tensor, n: int):
    for i in range(n):
        t = t[..., 1:] - t[..., :-1]
    return t


if __name__ == '__main__':
    # fig = plot_heat_map(np.random.randn(50, 10), np.eye(10)[np.random.randint(0, 9, (50, ))])
    # fig.show()
    # fig = plot_error_bar(np.random.randn(64, 15).clip(-1, 1), np.random.randn(64, 15).clip(-1, 1))
    # fig.show()
    # tvs = np.array([0, 0, 0, 0, 5.5, 2500, 500, 0.8, 0.89, 0.99, 0.81, 0.76, 1.05, 1.23, 0.01, 0.1])
    # norm_tvs = normalize_tvs(tvs)
    # print(norm_tvs)

    # torch.autograd.set_detect_anomaly(True)
    # z = torch.randint(0, 10, (800, ))
    # x = torch.randn(800, 5)
    # mean = torch.randn(10, 5, requires_grad=True)
    # std = torch.randn(10, 5, requires_grad=True)
    # targets = z.nonzero()[:, 0]  # index of target frame
    # res = torch.zeros(len(targets), 5)
    # for i, target in enumerate(targets):
    #     res[i] = (mean[z[target]] + torch.randn(1) * std[z[target]]).clamp(-1, 1)
    # frames = torch.linspace(0, len(z)-1, len(z))
    # f = Tinterp1d(targets, res)
    # res_ = f(frames)
    # print(res_)

    # from scipy.interpolate import interp1d
    # tar_np = targets.numpy()
    # res_np = res.detach().numpy()
    # print(tar_np.shape, res_np.shape)
    # f2 = interp1d(tar_np, res_np, kind='linear', axis=0)
    # res2_ = f2(frames.numpy())
    # print(res_ == torch.from_numpy(res2_))

    # f_pchi = TPchip(targets, res)
    # res_ = f_pchi(frames)

    # loss = ((x-res_)**2).mean()
    # loss.backward()
    #
    # print('mean.grad: {}\nstd.grad: {}'.format(mean.grad, std.grad))

    # res.detach_()
    # res_.detach_()
    # fig, axes = plt.subplots(figsize=(6, 30), nrows=15, ncols=1, sharex=True)
    # for i in range(15):
    #     axes[i].scatter(targets, res[:, i])
    #     axes[i].plot(frames, res_[:, i])
    #
    # fig.show()

    # ema_fp = '/data1/huangqinlong/PhonemeExtraction/data/mocha-timit/msak0_v1.1/msak0_023.ema'
    # read_a_ema_file(ema_fp, resample_rate=80)

    timit_dir = '/data1/huangqinlong/PhonemeExtraction/data/mocha-timit'
    save_ema_to_pickle(timit_dir, resample_rate=80, a=-1, b=1)

    # lab_fp = '/data1/huangqinlong/PhonemeExtraction/data/mocha-timit/msak0_v1.1/msak0_001.lab'
    # read_a_lab_file(lab_fp)

    # timit_dir = '/data1/huangqinlong/PhonemeExtraction/data/mocha-timit/fsew0_v1.1'
    # read_lab_path(timit_dir)
